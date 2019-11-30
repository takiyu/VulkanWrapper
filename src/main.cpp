#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
// Storage for dispatcher
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

template <typename T>
inline T clamp(const T& x, const T& min_v, const T& max_v) {
    return std::min(std::max(x, min_v), max_v);
}

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

template <typename Allocator>
static void PrintQueueFamilyProps(
        const std::vector<vk::QueueFamilyProperties, Allocator>& props) {
    std::cout << "QueueFamilyProperties" << std::endl;
    for (uint32_t i = 0; i < props.size(); i++) {
        const auto& flags_str = vk::to_string(props[i].queueFlags);
        std::cout << "  " << i << ": " << flags_str << std::endl;
    }
}

struct GlfwWinDeleter {
    void operator()(GLFWwindow* ptr) {
        glfwDestroyWindow(ptr);
    }
};

int main(int argc, char const* argv[]) {
    (void)argc, (void)argv;

    const char* app_name = "app name";
    const int app_version = 1;
    const char* engine_name = "engine name";
    const int engine_version = 1;
    uint32_t screen_w = 200;
    uint32_t screen_h = 200;

    // Create GLFW window
    glfwInit();
    atexit([]() { glfwTerminate(); });
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    std::unique_ptr<GLFWwindow, GlfwWinDeleter> window(glfwCreateWindow(
            static_cast<int>(screen_w), static_cast<int>(screen_h), app_name,
            nullptr, nullptr));
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("No Vulkan support");
    }
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
    vk::ApplicationInfo app_info = {app_name, app_version, engine_name,
                                    engine_version, VK_API_VERSION_1_1};
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

    // Create a window surface
    using Deleter = vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderDynamic>;
    vk::UniqueSurfaceKHR surface(
            [&]() {
                VkSurfaceKHR s = nullptr;
                VkResult err = glfwCreateWindowSurface(
                        instance.get(), window.get(), nullptr, &s);
                if (err) {
                    throw std::runtime_error("Failed to create window surface");
                }
                return s;
            }(),
            Deleter(instance.get()));
    //     vkCreateAndroidSurfaceKHR

    // Enumerate the physical devices
    auto physical_devices = instance->enumeratePhysicalDevices();
    vk::PhysicalDevice& physical_device = physical_devices.front();

    // Get queue family properties
    auto queue_family_props = physical_device.getQueueFamilyProperties();
    PrintQueueFamilyProps(queue_family_props);

    // Get the first index into queueFamiliyProperties which supports graphics
    uint32_t queue_family_idx = 0;
    for (auto&& prop : queue_family_props) {
        if (prop.queueFlags & vk::QueueFlagBits::eGraphics &&
            physical_device.getSurfaceSupportKHR(queue_family_idx,
                                                 surface.get())) {
            break;
        }
        queue_family_idx++;
    }
    if (queue_family_idx == queue_family_props.size()) {
        throw std::runtime_error("No sufficient queue");
    }

    // Create a logical device
    float queue_priority = 0.f;
    vk::DeviceQueueCreateInfo device_queue_create_info = {
            vk::DeviceQueueCreateFlags(), queue_family_idx, 1, &queue_priority};
    const std::vector<const char*> device_exts = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    vk::UniqueDevice device = physical_device.createDeviceUnique(
            {vk::DeviceCreateFlags(), 1, &device_queue_create_info, 0, nullptr,
             static_cast<uint32_t>(device_exts.size()), device_exts.data()});

    // Initialize dispatcher for device
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    // Create a command pool
    vk::UniqueCommandPool command_pool = device->createCommandPoolUnique(
            {vk::CommandPoolCreateFlags(), queue_family_idx});

    // Allocate a command buffer from the command pool
    const uint32_t N_COMMAND_BUFFERS = 1;
    auto command_buffers = device->allocateCommandBuffersUnique(
            {command_pool.get(), vk::CommandBufferLevel::ePrimary,
             N_COMMAND_BUFFERS});
    assert(command_buffers.size() == N_COMMAND_BUFFERS);

    // Use first command buffer
    vk::UniqueCommandBuffer& command_buffer = command_buffers[0];

    // Get the supported surface VkFormats
    auto surface_formats = physical_device.getSurfaceFormatsKHR(surface.get());
    assert(!surface_formats.empty());
    vk::Format surface_format = surface_formats[0].format;
    if (surface_format == vk::Format::eUndefined) {
        surface_format = vk::Format::eB8G8R8A8Unorm;
    }

    // Get the surface extent size
    vk::SurfaceCapabilitiesKHR surface_capas =
            physical_device.getSurfaceCapabilitiesKHR(surface.get());
    VkExtent2D swapchain_extent = surface_capas.currentExtent;
    if (swapchain_extent.width == std::numeric_limits<uint32_t>::max()) {
        // If the surface size is undefined, setting screen size is requested.
        const auto& min_ex = surface_capas.minImageExtent;
        const auto& max_ex = surface_capas.maxImageExtent;
        swapchain_extent.width = clamp(screen_w, min_ex.width, max_ex.width);
        swapchain_extent.height = clamp(screen_h, min_ex.height, max_ex.height);
    }

    // Set swapchain present mode
    vk::PresentModeKHR swapchain_present_mode = vk::PresentModeKHR::eFifo;

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

    // Create swapchain
    vk::UniqueSwapchainKHR swapchain = device->createSwapchainKHRUnique(
            {vk::SwapchainCreateFlagsKHR(), surface.get(),
             surface_capas.minImageCount, surface_format,
             vk::ColorSpaceKHR::eSrgbNonlinear, swapchain_extent, 1,
             vk::ImageUsageFlagBits::eColorAttachment,
             vk::SharingMode::eExclusive, 0, nullptr, pre_trans,
             composite_alpha, swapchain_present_mode, true, nullptr});

    // Get swapchain images
    std::vector<vk::Image> swapchain_imgs =
            device->getSwapchainImagesKHR(swapchain.get());

    // Create image views for swapchain
    std::vector<vk::UniqueImageView> img_views;
    img_views.reserve(swapchain_imgs.size());
    vk::ComponentMapping surface_comp_mapping(
            vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
            vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);
    vk::ImageSubresourceRange surface_sub_res_range(
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    for (auto image : swapchain_imgs) {
        auto img_view = device->createImageViewUnique(
                {vk::ImageViewCreateFlags(), image, vk::ImageViewType::e2D,
                 surface_format, surface_comp_mapping, surface_sub_res_range});
        img_views.push_back(std::move(img_view));
    }

    // Create depth image
    const vk::Format depth_format = vk::Format::eD16Unorm;
    auto depth_format_props = physical_device.getFormatProperties(depth_format);
    vk::ImageTiling tiling;
    if (depth_format_props.linearTilingFeatures &
        vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
        tiling = vk::ImageTiling::eLinear;
    } else if (depth_format_props.optimalTilingFeatures &
               vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
        tiling = vk::ImageTiling::eOptimal;
    } else {
        throw std::runtime_error(
                "DepthStencilAttachment is not supported for "
                "D16Unorm depth format.");
    }
    vk::UniqueImage depth_img = device->createImageUnique(
            {vk::ImageCreateFlags(), vk::ImageType::e2D, depth_format,
             vk::Extent3D(swapchain_extent, 1), 1, 1,
             vk::SampleCountFlagBits::e1, tiling,
             vk::ImageUsageFlagBits::eDepthStencilAttachment});

    // Allocate depth memory
    auto memory_props = physical_device.getMemoryProperties();
    auto memory_requs = device->getImageMemoryRequirements(depth_img.get());
    uint32_t type_bits = memory_requs.memoryTypeBits;
    uint32_t type_idx = uint32_t(~0);
    for (uint32_t i = 0; i < memory_props.memoryTypeCount; i++) {
        if ((type_bits & 1) && (memory_props.memoryTypes[i].propertyFlags ==
                                vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            type_idx = i;
            break;
        }
        type_bits >>= 1;
    }
    assert(type_idx != ~0);
    vk::UniqueDeviceMemory depth_memory =
            device->allocateMemoryUnique({memory_requs.size, type_idx});

    // Bind depth image and memory
    device->bindImageMemory(depth_img.get(), depth_memory.get(), 0);

    // Create depth view
    vk::ComponentMapping depth_comp_mapping(
            vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
            vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);
    vk::ImageSubresourceRange depth_sub_res_range(
            vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
    vk::UniqueImageView depth_view = device->createImageViewUnique(
            vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), depth_img.get(),
                                    vk::ImageViewType::e2D, depth_format,
                                    depth_comp_mapping, depth_sub_res_range));

    // Create uniform buffer
    glm::mat4x4 model_mat = glm::mat4x4(1.0f);
    glm::mat4x4 view_mat = glm::lookAt(glm::vec3(-5.0f, 3.0f, -10.0f),
                                       glm::vec3(0.0f, 0.0f, 0.0f),
                                       glm::vec3(0.0f, -1.0f, 0.0f));
    glm::mat4x4 proj_mat =
            glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    glm::mat4x4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                            0.0f, 0.0f, 0.5f, 1.0f};  // vulkan clip space has
                                                      // inverted y and half z !
    glm::mat4x4 mvpc_mat = clip_mat * proj_mat * view_mat * model_mat;

    vk::UniqueBuffer uniform_buffer = device->createBufferUnique(
            vk::BufferCreateInfo(vk::BufferCreateFlags(), sizeof(mvpc_mat),
                                 vk::BufferUsageFlagBits::eUniformBuffer));

//     vk::MemoryRequirements uniform_memory_requs =
//             device->getBufferMemoryRequirements(uniform_buffer.get());
//     uint32_t typeIndex = vk::su::findMemoryType(
//             physicalDevice.getMemoryProperties(),
//             memoryRequirements.memoryTypeBits,
//             vk::MemoryPropertyFlagBits::eHostVisible |
//                     vk::MemoryPropertyFlagBits::eHostCoherent);
//     vk::UniqueDeviceMemory uniformDataMemory = device->allocateMemoryUnique(
//             vk::MemoryAllocateInfo(memoryRequirements.size, typeIndex));
//
//     uint8_t* pData = static_cast<uint8_t*>(device->mapMemory(
//             uniformDataMemory.get(), 0, memoryRequirements.size));
//     memcpy(pData, &mvpc_mat, sizeof(mvpc_mat));
//     device->unmapMemory(uniformDataMemory.get());
//
//     device->bindBufferMemory(uniform_buffer.get(), uniformDataMemory.get(),
//                              0);

    while (!glfwWindowShouldClose(window.get())) {
        glfwPollEvents();
        break;
    }

    std::cout << "exit" << std::endl;

    return 0;
}
