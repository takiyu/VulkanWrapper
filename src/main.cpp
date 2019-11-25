#include <bits/stdint-uintn.h>

#include <iostream>
#include <stdexcept>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
// Storage for dispatcher
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

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

    // Create GLFW window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    std::unique_ptr<GLFWwindow, GlfwWinDeleter> window(
            glfwCreateWindow(100, 100, app_name, nullptr, nullptr));
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("No Vulkan support");
    }
    // Print extension names required by GLFW
    uint32_t cnt = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&cnt);
    for (size_t i = 0; i < cnt; i++) {
        std::cout << exts[i] << std::endl;
    }

    // Initialize dispatcher with `vkGetInstanceProcAddr`, to get the instance
    // independent function pointers
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
            vk::DynamicLoader().getProcAddress<PFN_vkGetInstanceProcAddr>(
                    "vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Create a Vulkan instance
    std::vector<char const*> enabled_layer = {"VK_LAYER_KHRONOS_validation"};
    std::vector<char const*> enabled_exts = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
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

    // Enumerate the physical devices
    auto physical_devices = instance->enumeratePhysicalDevices();
    vk::PhysicalDevice& physical_device = physical_devices.front();

    // Get queue family properties
    auto queue_family_props = physical_device.getQueueFamilyProperties();
    PrintQueueFamilyProps(queue_family_props);

    // Get the first index into queueFamiliyProperties which supports graphics
    uint32_t queue_family_idx = 0;
    for (auto&& prop : queue_family_props) {
        if (prop.queueFlags & vk::QueueFlagBits::eGraphics) {
            break;
        }
        queue_family_idx++;
    }
    if (queue_family_idx == queue_family_props.size()) {
        throw std::runtime_error("");
    }

    // Create a logical device
    float queue_priority = 0.f;
    vk::DeviceQueueCreateInfo device_queue_create_info = {
            vk::DeviceQueueCreateFlags(), queue_family_idx, 1, &queue_priority};
    vk::UniqueDevice device = physical_device.createDeviceUnique(
            {vk::DeviceCreateFlags(), 1, &device_queue_create_info});

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

    // Create a window surface
    vk::UniqueSurfaceKHR surface([&]() {
        VkSurfaceKHR s;
        VkResult err = glfwCreateWindowSurface(instance.get(), window.get(),
                                               nullptr, &s);
        std::cout << err << std::endl;
        if (err) {
            throw std::runtime_error("Failed to create window surface");
        }
        return s;
    }());
    //         instance->createWin32SurfaceKHRUnique({vk::Win32SurfaceCreateFlagsKHR(),
    //         GetModuleHandle(nullptr), window});
    //             instance->createXlibSurfaceKHRUnique;

    //     vkCreateAndroidSurfaceKHR

    //     vk::UniqueSurfaceKHR surface =
    //     instance->createWin32SurfaceKHRUnique(vk::Win32SurfaceCreateInfoKHR(vk::Win32SurfaceCreateFlagsKHR(),
    //     GetModuleHandle(nullptr), window));

    //     while (!glfwWindowShouldClose(window)) {
    //         glfwPollEvents();
    //     }
    //
    //     glfwDestroyWindow(window);
    //     glfwTerminate();

    return 0;
}
