#include "vkw.h"

#include <bits/stdint-uintn.h>

#include <iostream>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

// Storage for dispatcher
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace vkw {

namespace {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
template <typename T>
inline T clamp(const T& x, const T& min_v, const T& max_v) {
    return std::min(std::max(x, min_v), max_v);
}

template <typename BitType, typename MaskType = VkFlags>
bool IsSufficient(const vk::Flags<BitType, MaskType>& actual_flags,
                  const vk::Flags<BitType, MaskType>& require_flags) {
    return (actual_flags & require_flags) == require_flags;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static VkBool32 DebugMessengerCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
        VkDebugUtilsMessageTypeFlagsEXT msg_types,
        VkDebugUtilsMessengerCallbackDataEXT const* callback, void*) {
    // Create corresponding strings
    const std::string& severity_str =
            vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(
                    msg_severity));
    const std::string& type_str = vk::to_string(
            static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(msg_types));

    // Print messages
    std::cerr << "-----------------------------------------------" << std::endl;
    std::cerr << severity_str << ": " << type_str << ":" << std::endl;
    std::cerr << "  Message ID Name (number) = <" << callback->pMessageIdName
              << "> (" << callback->messageIdNumber << ")" << std::endl;
    std::cerr << "  Message = \"" << callback->pMessage << "\"" << std::endl;
    if (0 < callback->queueLabelCount) {
        std::cerr << "  Queue Labels:" << std::endl;
        for (uint8_t i = 0; i < callback->queueLabelCount; i++) {
            const auto& name = callback->pQueueLabels[i].pLabelName;
            std::cerr << "    " << i << ": " << name << std::endl;
        }
    }
    if (0 < callback->cmdBufLabelCount) {
        std::cerr << "  CommandBuffer Labels:" << std::endl;
        for (uint8_t i = 0; i < callback->cmdBufLabelCount; i++) {
            const auto& name = callback->pCmdBufLabels[i].pLabelName;
            std::cerr << "    " << i << ": " << name << std::endl;
        }
    }
    if (0 < callback->objectCount) {
        std::cerr << "  Objects:" << std::endl;
        for (uint8_t i = 0; i < callback->objectCount; i++) {
            const auto& type = vk::to_string(static_cast<vk::ObjectType>(
                    callback->pObjects[i].objectType));
            const auto& handle = callback->pObjects[i].objectHandle;
            std::cerr << "    " << static_cast<int>(i) << ":" << std::endl;
            std::cerr << "      objectType   = " << type << std::endl;
            std::cerr << "      objectHandle = " << handle << std::endl;
            if (callback->pObjects[i].pObjectName) {
                const auto& on = callback->pObjects[i].pObjectName;
                std::cerr << "      objectName   = <" << on << ">" << std::endl;
            }
        }
    }
    std::cerr << "-----------------------------------------------" << std::endl;
    return VK_TRUE;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static vk::Format GetSurfaceFormat(const vk::PhysicalDevice& physical_device,
                                   const vk::SurfaceKHR& surface) {
    auto surface_formats = physical_device.getSurfaceFormatsKHR(surface);
    assert(!surface_formats.empty());
    vk::Format surface_format = surface_formats[0].format;
    if (surface_format == vk::Format::eUndefined) {
        surface_format = vk::Format::eB8G8R8A8Unorm;
    }
    return surface_format;
}

static auto SelectSwapchainProps(const vk::PhysicalDevice& physical_device,
                                 const vk::SurfaceKHR& surface, uint32_t win_w,
                                 uint32_t win_h) {
    // Get all capabilities
    const vk::SurfaceCapabilitiesKHR& surface_capas =
            physical_device.getSurfaceCapabilitiesKHR(surface);

    // Get the surface extent size
    VkExtent2D swapchain_extent = surface_capas.currentExtent;
    if (swapchain_extent.width == std::numeric_limits<uint32_t>::max()) {
        // If the surface size is undefined, setting screen size is requested.
        const auto& min_ex = surface_capas.minImageExtent;
        const auto& max_ex = surface_capas.maxImageExtent;
        swapchain_extent.width = clamp(win_w, min_ex.width, max_ex.width);
        swapchain_extent.height = clamp(win_h, min_ex.height, max_ex.height);
    }

    // Set swapchain pre-transform
    vk::SurfaceTransformFlagBitsKHR pre_trans = surface_capas.currentTransform;
    if (surface_capas.supportedTransforms &
        vk::SurfaceTransformFlagBitsKHR::eIdentity) {
        pre_trans = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    }

    // Set swapchain composite alpha
    vk::CompositeAlphaFlagBitsKHR composite_alpha;
    const auto& suppored_flag = surface_capas.supportedCompositeAlpha;
    if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
    } else if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
    } else if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::eInherit) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
    } else {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    }

    const uint32_t min_img_cnt = surface_capas.minImageCount;

    return std::make_tuple(swapchain_extent, pre_trans, composite_alpha,
                           min_img_cnt);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
vk::UniqueDeviceMemory AllocMemory(
        const vk::Device& device, const vk::PhysicalDevice& physical_device,
        const vk::MemoryRequirements& memory_requs,
        const vk::MemoryPropertyFlags& require_flags) {
    auto actual_memory_props = physical_device.getMemoryProperties();
    uint32_t type_bits = memory_requs.memoryTypeBits;
    uint32_t type_idx = uint32_t(~0);
    for (uint32_t i = 0; i < actual_memory_props.memoryTypeCount; i++) {
        if ((type_bits & 1) &&
            IsSufficient(actual_memory_props.memoryTypes[i].propertyFlags,
                         require_flags)) {
            type_idx = i;
            break;
        }
        type_bits >>= 1;
    }
    if (type_idx == uint32_t(~0)) {
        throw std::runtime_error("Failed to allocate requested memory");
    }

    return device.allocateMemoryUnique({memory_requs.size, type_idx});
}

static vk::UniqueImageView CreateImageView(const vk::Image& img,
                                           const vk::Format& format,
                                           const vk::ImageAspectFlags& aspects,
                                           const vk::Device& device) {
    const vk::ComponentMapping comp_mapping(
            vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
            vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);
    vk::ImageSubresourceRange sub_res_range(aspects, 0, 1, 0, 1);
    auto view = device.createImageViewUnique({vk::ImageViewCreateFlags(), img,
                                              vk::ImageViewType::e2D, format,
                                              comp_mapping, sub_res_range});
    return view;
}

static void SendToDevice(const vk::Device& device,
                         const vk::DeviceMemory& dev_mem,
                         const vk::DeviceSize& dev_mem_size, const void* data,
                         uint64_t n_bytes) {
    uint8_t* dev_p = static_cast<uint8_t*>(device.mapMemory(dev_mem, 0, dev_mem_size));
    memcpy(dev_p, data, n_bytes);
    device.unmapMemory(dev_mem);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

}  // namespace

// -----------------------------------------------------------------------------
// ----------------------------------- GLFW ------------------------------------
// -----------------------------------------------------------------------------
GLFWWindowUnique InitGLFWWindow(const std::string& win_name, uint32_t win_w,
                                uint32_t win_h) {
    // Initialize GLFW
    glfwInit();
    atexit([]() { glfwTerminate(); });

    // Create GLFW window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWWindowUnique window(
            glfwCreateWindow(static_cast<int>(win_w), static_cast<int>(win_h),
                             win_name.c_str(), nullptr, nullptr));
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("No Vulkan support");
    }
    return window;
}

// -----------------------------------------------------------------------------
// --------------------------------- Instance ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueInstance CreateInstance(const std::string& app_name,
                                  uint32_t app_version,
                                  const std::string& engine_name,
                                  uint32_t engine_version) {
    // Print extension names required by GLFW
    uint32_t n_glfw_exts = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&n_glfw_exts);

    // Initialize dispatcher with `vkGetInstanceProcAddr`, to get the instance
    // independent function pointers
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
            vk::DynamicLoader().getProcAddress<PFN_vkGetInstanceProcAddr>(
                    "vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Create a Vulkan instance
    std::vector<char const*> enabled_layer = {"VK_LAYER_KHRONOS_validation"};
    std::vector<char const*> enabled_exts = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    for (uint32_t i = 0; i < n_glfw_exts; i++) {
        enabled_exts.push_back(glfw_exts[i]);
    }
    vk::ApplicationInfo app_info = {app_name.c_str(), app_version,
                                    engine_name.c_str(), engine_version,
                                    VK_API_VERSION_1_1};
    vk::UniqueInstance instance = vk::createInstanceUnique(
            {vk::InstanceCreateFlags(), &app_info,
             static_cast<uint32_t>(enabled_layer.size()), enabled_layer.data(),
             static_cast<uint32_t>(enabled_exts.size()), enabled_exts.data()});

    // Initialize dispatcher with Instance to get all the other function ptrs.
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    // Create debug messenger
    vk::UniqueDebugUtilsMessengerEXT debug_messenger =
            instance->createDebugUtilsMessengerEXTUnique(
                    {{},
                     {vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError},
                     {vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation},
                     &DebugMessengerCallback});

    return instance;
}

// -----------------------------------------------------------------------------
// ------------------------------ PhysicalDevice -------------------------------
// -----------------------------------------------------------------------------
std::vector<vk::PhysicalDevice> GetPhysicalDevices(const vk::Instance& inst) {
    return inst.enumeratePhysicalDevices();
}

// -----------------------------------------------------------------------------
// ---------------------------------- Surface ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueSurfaceKHR CreateSurface(const vk::Instance& instance,
                                   GLFWwindow* window) {
    // Create a window surface (GLFW)
    VkSurfaceKHR s_raw = nullptr;
    VkResult err = glfwCreateWindowSurface(instance, window, nullptr, &s_raw);
    if (err) {
        throw std::runtime_error("Failed to create window surface");
    }

    // TODO: Android (non-GLFW) version
    // vkCreateAndroidSurfaceKHR

    // Smart wrapper
    using Deleter = vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderDynamic>;
    vk::UniqueSurfaceKHR surface(s_raw, Deleter(instance));

    return surface;
}

// -----------------------------------------------------------------------------
// -------------------------------- Queue Family -------------------------------
// -----------------------------------------------------------------------------
void PrintQueueFamilyProps(const vk::PhysicalDevice& physical_device) {
    const auto& props = physical_device.getQueueFamilyProperties();

    std::cout << "QueueFamilyProperties" << std::endl;
    for (uint32_t i = 0; i < props.size(); i++) {
        const auto prop = props[i];
        const auto& flags_str = vk::to_string(prop.queueFlags);
        const auto& max_queue_cnt = prop.queueCount;
        std::cout << "  " << i << ": " << flags_str
                  << "  (max_cnt:" << max_queue_cnt << ")" << std::endl;
    }
}

std::vector<uint32_t> GetQueueFamilyIdxs(
        const vk::PhysicalDevice& physical_device,
        const vk::QueueFlags& queue_flags) {
    const auto& props = physical_device.getQueueFamilyProperties();

    // Search sufficient queue family indices
    std::vector<uint32_t> queue_family_idxs;
    for (uint32_t i = 0; i < props.size(); i++) {
        if (IsSufficient(props[i].queueFlags, queue_flags)) {
            queue_family_idxs.push_back(i);
        }
    }
    return queue_family_idxs;
}

uint32_t GetGraphicPresentQueueFamilyIdx(
        const vk::PhysicalDevice& physical_device,
        const vk::SurfaceKHR& surface, const vk::QueueFlags& queue_flags) {
    // Get graphics queue family indices
    auto graphic_idxs = GetQueueFamilyIdxs(physical_device, queue_flags);

    // Extract present queue indices
    std::vector<uint32_t> graphic_present_idxs;
    for (auto&& idx : graphic_idxs) {
        if (physical_device.getSurfaceSupportKHR(idx, surface)) {
            graphic_present_idxs.push_back(idx);
        }
    }

    // Check status
    if (graphic_present_idxs.size() == 0) {
        throw std::runtime_error("No sufficient queue for graphic present");
    }
    // Return the first index
    return graphic_present_idxs.front();
}

// -----------------------------------------------------------------------------
// ----------------------------------- Device ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueDevice CreateDevice(uint32_t queue_family_idx,
                              const vk::PhysicalDevice& physical_device,
                              uint32_t n_queues, bool swapchain_support) {
    // Create queue create info
    std::vector<float> queue_priorites(n_queues, 0.f);
    vk::DeviceQueueCreateInfo device_queue_create_info = {
            vk::DeviceQueueCreateFlags(), queue_family_idx, n_queues,
            queue_priorites.data()};

    // Create device extension strings
    std::vector<const char*> device_exts;
    if (swapchain_support) {
        device_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    // Create a logical device
    vk::UniqueDevice device = physical_device.createDeviceUnique(
            {vk::DeviceCreateFlags(), 1, &device_queue_create_info, 0, nullptr,
             static_cast<uint32_t>(device_exts.size()), device_exts.data()});

    // Initialize dispatcher for device
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    return device;
}

// -----------------------------------------------------------------------------
// ------------------------------- Command Buffer ------------------------------
// -----------------------------------------------------------------------------
CommandBuffersPack CreateCommandBuffers(const vk::Device& device,
                                        uint32_t queue_family_idx,
                                        uint32_t n_cmd_buffers) {
    // Create a command pool
    vk::UniqueCommandPool command_pool = device.createCommandPoolUnique(
            {vk::CommandPoolCreateFlags(), queue_family_idx});

    // Allocate a command buffer from the command pool
    auto cmd_bufs = device.allocateCommandBuffersUnique(
            {command_pool.get(), vk::CommandBufferLevel::ePrimary,
             n_cmd_buffers});

    return {std::move(command_pool), std::move(cmd_bufs)};
}

// -----------------------------------------------------------------------------
// --------------------------------- Swapchain ---------------------------------
// -----------------------------------------------------------------------------
SwapchainPack CreateSwapchain(const vk::PhysicalDevice& physical_device,
                              const vk::Device& device,
                              const vk::SurfaceKHR& surface, uint32_t win_w,
                              uint32_t win_h) {
    // Set swapchain present mode
    const vk::PresentModeKHR swapchain_present_mode = vk::PresentModeKHR::eFifo;

    // Get the supported surface VkFormats
    const auto surface_format = GetSurfaceFormat(physical_device, surface);

    // Select properties from capabilities
    auto props = SelectSwapchainProps(physical_device, surface, win_w, win_h);
    const auto& swapchain_extent = std::get<0>(props);
    const auto& pre_trans = std::get<1>(props);
    const auto& composite_alpha = std::get<2>(props);
    const auto& min_img_cnt = std::get<3>(props);

    // Create swapchain
    vk::UniqueSwapchainKHR swapchain = device.createSwapchainKHRUnique(
            {vk::SwapchainCreateFlagsKHR(), surface, min_img_cnt,
             surface_format, vk::ColorSpaceKHR::eSrgbNonlinear,
             swapchain_extent, 1, vk::ImageUsageFlagBits::eColorAttachment,
             vk::SharingMode::eExclusive, 0, nullptr, pre_trans,
             composite_alpha, swapchain_present_mode, true, nullptr});

    // Create image views
    auto swapchain_imgs = device.getSwapchainImagesKHR(swapchain.get());
    std::vector<vk::UniqueImageView> img_views;
    img_views.reserve(swapchain_imgs.size());
    for (auto img : swapchain_imgs) {
        auto img_view = CreateImageView(
                img, surface_format, vk::ImageAspectFlagBits::eColor, device);
        img_views.push_back(std::move(img_view));
    }

    return {std::move(swapchain), std::move(img_views), swapchain_extent};
}

// -----------------------------------------------------------------------------
// ----------------------------------- Image -----------------------------------
// -----------------------------------------------------------------------------
ImagePack CreateImage(const vk::PhysicalDevice& physical_device,
                      const vk::Device& device, const vk::Format format,
                      const vk::Extent2D& size,
                      const vk::ImageUsageFlags& usage,
                      const vk::MemoryPropertyFlags& memory_props,
                      const vk::ImageAspectFlags& aspects, bool is_staging,
                      bool is_shared) {
    // Select tiling mode
    vk::ImageTiling tiling =
            is_staging ? vk::ImageTiling::eOptimal : vk::ImageTiling::eLinear;
    // Select sharing mode
    vk::SharingMode sharing = is_shared ? vk::SharingMode::eConcurrent :
                                          vk::SharingMode::eExclusive;

    // Create image
    auto img = device.createImageUnique(
            {vk::ImageCreateFlags(), vk::ImageType::e2D, format,
             vk::Extent3D(size, 1), 1, 1, vk::SampleCountFlagBits::e1, tiling,
             usage, sharing});

    // Allocate memory
    auto memory_requs = device.getImageMemoryRequirements(*img);
    auto device_mem =
            AllocMemory(device, physical_device, memory_requs, memory_props);

    // Bind memory
    device.bindImageMemory(img.get(), device_mem.get(), 0);

    // Create image view
    auto img_view = CreateImageView(*img, format, aspects, device);

    return {std::move(img), std::move(img_view), std::move(device_mem),
            memory_requs.size};
}

void SendToDevice(const vk::Device& device, const ImagePack& img_pack,
                  const void* data, uint64_t n_bytes) {
    SendToDevice(device, *img_pack.dev_mem, img_pack.dev_mem_size, data,
                 n_bytes);
}

// -----------------------------------------------------------------------------
// ----------------------------------- Buffer ----------------------------------
// -----------------------------------------------------------------------------
BufferPack CreateBuffer(const vk::PhysicalDevice& physical_device,
                        const vk::Device& device, const vk::DeviceSize& size,
                        const vk::BufferUsageFlags& usage,
                        const vk::MemoryPropertyFlags& memory_props) {
    // Create buffer
    auto buf =
            device.createBufferUnique({vk::BufferCreateFlags(), size, usage});

    // Allocate memory
    auto memory_requs = device.getBufferMemoryRequirements(*buf);
    auto device_mem =
            AllocMemory(device, physical_device, memory_requs, memory_props);

    // Bind memory
    device.bindBufferMemory(buf.get(), device_mem.get(), 0);

    return {std::move(buf), std::move(device_mem), memory_requs.size};
}

void SendToDevice(const vk::Device& device, const BufferPack& buf_pack,
                  const void* data, uint64_t n_bytes) {
    SendToDevice(device, *(buf_pack.dev_mem), buf_pack.dev_mem_size, data,
                 n_bytes);
}

}  // namespace vkw
