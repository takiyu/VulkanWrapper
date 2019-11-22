#include <iostream>
#include <stdexcept>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
// Storage for dispatcher
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

VkBool32 debugUtilsMessengerCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
        void* /*pUserData*/) {
    std::cerr << vk::to_string(
                         static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(
                                 messageSeverity))
              << ": "
              << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(
                         messageTypes))
              << ":\n";
    std::cerr << "\t"
              << "messageIDName   = <" << pCallbackData->pMessageIdName
              << ">\n";
    std::cerr << "\t"
              << "messageIdNumber = " << pCallbackData->messageIdNumber << "\n";
    std::cerr << "\t"
              << "message         = <" << pCallbackData->pMessage << ">\n";
    if (0 < pCallbackData->queueLabelCount) {
        std::cerr << "\t"
                  << "Queue Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->queueLabelCount; i++) {
            std::cerr << "\t\t"
                      << "lableName = <"
                      << pCallbackData->pQueueLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->cmdBufLabelCount) {
        std::cerr << "\t"
                  << "CommandBuffer Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->cmdBufLabelCount; i++) {
            std::cerr << "\t\t"
                      << "labelName = <"
                      << pCallbackData->pCmdBufLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->objectCount) {
        std::cerr << "\t"
                  << "Objects:\n";
        for (uint8_t i = 0; i < pCallbackData->objectCount; i++) {
            std::cerr << "\t\t"
                      << "Object " << i << "\n";
            std::cerr << "\t\t\t"
                      << "objectType   = "
                      << vk::to_string(static_cast<vk::ObjectType>(
                                 pCallbackData->pObjects[i].objectType))
                      << "\n";
            std::cerr << "\t\t\t"
                      << "objectHandle = "
                      << pCallbackData->pObjects[i].objectHandle << "\n";
            if (pCallbackData->pObjects[i].pObjectName) {
                std::cerr << "\t\t\t"
                          << "objectName   = <"
                          << pCallbackData->pObjects[i].pObjectName << ">\n";
            }
        }
    }
    return VK_TRUE;
}

int main(int argc, char const* argv[]) {
    const char* app_name = "app name";
    const int app_version = 1;
    const char* engine_name = "engine name";
    const int engine_version = 1;

    // Create GLFW window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(100, 100, app_name, nullptr, nullptr);
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("No Vulkan support");
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

    instance->createDebugUtilsMessengerEXTUnique(
            vk::DebugUtilsMessengerCreateInfoEXT(
                    {},
                    {vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                     vk::DebugUtilsMessageSeverityFlagBitsEXT::eError},
                    {vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                     vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                     vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation},
                    &debugUtilsMessengerCallback));

    // Enumerate the physical devices
    auto physical_devices = instance->enumeratePhysicalDevices();
    vk::PhysicalDevice& physical_device = physical_devices.front();

    // Get queue family properties
    auto queue_family_props = physical_device.getQueueFamilyProperties();

    // get the first index into queueFamiliyProperties which supports graphics
    const uint32_t queue_family_idx = static_cast<uint32_t>(std::distance(
            queue_family_props.begin(),
            std::find_if(queue_family_props.begin(), queue_family_props.end(),
                         [](vk::QueueFamilyProperties const& qfp) {
                             return qfp.queueFlags &
                                    vk::QueueFlagBits::eGraphics;
                         })));

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
    auto a = device->allocateCommandBuffersUnique(
                                    {command_pool.get(),
                                     vk::CommandBufferLevel::ePrimary, 1});
    std::cout << a.size() << std::endl;
//     vk::UniqueCommandBuffer command_buffer =
//             std::move(device->allocateCommandBuffersUnique(
//                                     {command_pool.get(),
//                                      vk::CommandBufferLevel::ePrimary, 1})
//                               .front());

    //     instance->createXlibSurfaceKHRUnique;

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
