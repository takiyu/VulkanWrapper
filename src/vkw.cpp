#include <vkw/vkw.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace vkw {

namespace {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
const std::string ENGINE_NAME = "VKW";
const uint32_t ENGINE_VERSION = 0;

vk::PhysicalDevice GetFirstPhysicalDevice(const vk::UniqueInstance& instance) {
    // Get a physical device
    auto physical_devices = vkw::GetPhysicalDevices(instance);
    const size_t n_phy_devices = physical_devices.size();
    if (n_phy_devices == 0) {
        throw std::runtime_error("No physical devices");
    }
    if (1 < n_phy_devices) {
        std::stringstream ss;
        ss << "Non single physical deivces (" << n_phy_devices << "), ";
        ss << "Using first one.";
        vkw::PrintInfo(ss.str());
    }
    return physical_devices.front();
}

auto InitWithoutDisplay(const std::string& app_name, uint32_t app_version,
                        uint32_t n_queues, bool debug_enable) {
    const bool display_enable = false;

    // Create instance
    vk::UniqueInstance instance =
            vkw::CreateInstance(app_name, app_version, ENGINE_NAME,
                                ENGINE_VERSION, debug_enable, display_enable);
    // Get a physical_device
    vk::PhysicalDevice physical_device = GetFirstPhysicalDevice(instance);

    // Select queue family
    auto queue_family_idxs = GetQueueFamilyIdxs(physical_device);
    if (queue_family_idxs.empty()) {
        throw std::runtime_error("No sufficient queue for graphics");
    }
    const uint32_t queue_family_idx = queue_family_idxs.front();
    // Create device
    vk::UniqueDevice device = vkw::CreateDevice(
            queue_family_idx, physical_device, n_queues, display_enable);

    // Get queues
    std::vector<vk::Queue> queues;
    queues.reserve(n_queues);
    for (uint32_t i = 0; i < n_queues; i++) {
        queues.push_back(vkw::GetQueue(device, queue_family_idx, i));
    }

    return std::make_tuple(std::move(instance), physical_device,
                           std::move(device), std::move(queues),
                           queue_family_idx);
}

template <typename T>
auto InitWithDisplay(const std::string& app_name, uint32_t app_version,
                     uint32_t n_queues, bool debug_enable, const T& window) {
    const bool display_enable = true;

    // Create instance
    vk::UniqueInstance instance =
            vkw::CreateInstance(app_name, app_version, ENGINE_NAME,
                                ENGINE_VERSION, debug_enable, display_enable);
    // Get a physical_device
    vk::PhysicalDevice physical_device = GetFirstPhysicalDevice(instance);

    // Create surface
    vk::UniqueSurfaceKHR surface = vkw::CreateSurface(instance, window);
    vk::Format surface_format = vkw::GetSurfaceFormat(physical_device, surface);

    // Select queue family
    const uint32_t queue_family_idx =
            vkw::GetGraphicPresentQueueFamilyIdx(physical_device, surface);
    // Create device
    vk::UniqueDevice device = vkw::CreateDevice(
            queue_family_idx, physical_device, n_queues, display_enable);

    // Create swapchain
    vkw::SwapchainPackPtr swapchain_pack =
            vkw::CreateSwapchainPack(physical_device, device, surface);

    // Get queues
    std::vector<vk::Queue> queues;
    queues.reserve(n_queues);
    for (uint32_t i = 0; i < n_queues; i++) {
        queues.push_back(vkw::GetQueue(device, queue_family_idx, i));
    }

    return std::make_tuple(std::move(instance), physical_device,
                           std::move(device), std::move(surface),
                           std::move(swapchain_pack), std::move(queues),
                           queue_family_idx, surface_format);
}
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

}  // namespace

// -----------------------------------------------------------------------------
// ------------------------------ Graphics Context -----------------------------
// -----------------------------------------------------------------------------
GraphicsContext::GraphicsContext(const std::string& app_name,
                                 uint32_t app_version, uint32_t n_queues,
                                 bool debug_enable) {
    // Initialize with display environment
    std::tie(m_instance, m_physical_device, m_device, m_queues,
             m_queue_family_idx) =
            InitWithoutDisplay(app_name, app_version, n_queues, debug_enable);
}

GraphicsContext::GraphicsContext(const std::string& app_name,
                                 uint32_t app_version, uint32_t n_queues,
                                 const vkw::WindowPtr& window,
                                 bool debug_enable) {
    // Set dependent variable
    m_window = window;

    // Initialize with display environment
    std::tie(m_instance, m_physical_device, m_device, m_surface,
             m_swapchain_pack, m_queues, m_queue_family_idx, m_surface_format) =
            InitWithDisplay(app_name, app_version, n_queues, debug_enable,
                            window);
}

const WindowPtr& GraphicsContext::getWindow() const {
    return m_window;
}

const vk::UniqueInstance& GraphicsContext::getInstance() const {
    return m_instance;
}

const vk::PhysicalDevice& GraphicsContext::getPhysicalDevice() const {
    return m_physical_device;
}

const vk::UniqueDevice& GraphicsContext::getDevice() const {
    return m_device;
}

const vk::UniqueSurfaceKHR& GraphicsContext::getSurface() const {
    return m_surface;
}

const SwapchainPackPtr& GraphicsContext::getSwapchainPack() const {
    return m_swapchain_pack;
}

const std::vector<vk::Queue>& GraphicsContext::getQueues() const {
    return m_queues;
}

uint32_t GraphicsContext::getQueueFamilyIdx() const {
    return m_queue_family_idx;
}

vk::Format GraphicsContext::getSurfaceFormat() const {
    return m_surface_format;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

}  // namespace vkw
