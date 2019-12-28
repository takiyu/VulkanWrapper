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
    const bool display_enable = false;

    // Create instance
    m_instance =
            vkw::CreateInstance(app_name, app_version, ENGINE_NAME,
                                ENGINE_VERSION, debug_enable, display_enable);
    // Get a physical_device
    m_physical_device = GetFirstPhysicalDevice(m_instance);

    // Select queue family
    auto queue_family_idxs = GetQueueFamilyIdxs(m_physical_device);
    if (queue_family_idxs.empty()) {
        throw std::runtime_error("No sufficient queue for graphics");
    }
    m_queue_family_idx = queue_family_idxs.front();
    // Create device
    m_device = vkw::CreateDevice(m_queue_family_idx, m_physical_device,
                                 n_queues, display_enable);

    // Get queues
    m_queues.clear();
    m_queues.reserve(n_queues);
    for (uint32_t i = 0; i < n_queues; i++) {
        m_queues.push_back(vkw::GetQueue(m_device, m_queue_family_idx, i));
    }
}

GraphicsContext::GraphicsContext(const std::string& app_name,
                                 uint32_t app_version, uint32_t n_queues,
                                 const vkw::WindowPtr& window,
                                 bool debug_enable) {
    // Initialize with display environment
    const bool display_enable = true;

    // Set dependent variable
    m_window = window;

    // Create instance
    m_instance =
            vkw::CreateInstance(app_name, app_version, ENGINE_NAME,
                                ENGINE_VERSION, debug_enable, display_enable);
    // Get a physical_device
    m_physical_device = GetFirstPhysicalDevice(m_instance);

    // Create surface
    m_surface = vkw::CreateSurface(m_instance, m_window);
    m_surface_format = vkw::GetSurfaceFormat(m_physical_device, m_surface);

    // Select queue family
    m_queue_family_idx =
            vkw::GetGraphicPresentQueueFamilyIdx(m_physical_device, m_surface);
    // Create device
    m_device = vkw::CreateDevice(m_queue_family_idx, m_physical_device,
                                 n_queues, display_enable);

    // Create swapchain
    m_swapchain_pack =
            vkw::CreateSwapchainPack(m_physical_device, m_device, m_surface);

    // Get queues
    m_queues.clear();
    m_queues.reserve(n_queues);
    for (uint32_t i = 0; i < n_queues; i++) {
        m_queues.push_back(vkw::GetQueue(m_device, m_queue_family_idx, i));
    }
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
// ----------------------------------- Image -----------------------------------
// -----------------------------------------------------------------------------
Image::Image(const GraphicsContextPtr& context, const vk::Format& format,
             const vk::Extent2D& size, const vk::ImageUsageFlags& usage,
             const vk::MemoryPropertyFlags& memory_props,
             const vk::ImageAspectFlags& aspects, bool is_staging,
             bool is_shared) {
    // Set dependent variable
    m_context = context;

    // Create image pack
    m_img_pack = vkw::CreateImagePack(
            m_context->getPhysicalDevice(), m_context->getDevice(), format,
            size, usage, memory_props, aspects, is_staging, is_shared);
}

void Image::sendToDevice(const void* data, uint64_t n_bytes) {
    SendToDevice(m_context->getDevice(), m_img_pack, data, n_bytes);
}

const vkw::ImagePackPtr& Image::getImagePack() const {
    return m_img_pack;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

}  // namespace vkw
