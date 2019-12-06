#ifndef VKW_H_20191130
#define VKW_H_20191130

#include <bits/stdint-uintn.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#include <vulkan/vulkan.hpp>

namespace vkw {

// -----------------------------------------------------------------------------
// ----------------------------------- GLFW ------------------------------------
// -----------------------------------------------------------------------------
struct GLFWWindowDeleter {
    void operator()(GLFWwindow* ptr) {
        glfwDestroyWindow(ptr);
    }
};
using GLFWWindowUnique = std::unique_ptr<GLFWwindow, GLFWWindowDeleter>;

GLFWWindowUnique InitGLFWWindow(const std::string& win_name, uint32_t win_w,
                                uint32_t win_h);

// -----------------------------------------------------------------------------
// --------------------------------- Instance ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueInstance CreateInstance(const std::string& app_name,
                                  uint32_t app_version,
                                  const std::string& engine_name,
                                  uint32_t engine_version);

// -----------------------------------------------------------------------------
// ------------------------------ PhysicalDevice -------------------------------
// -----------------------------------------------------------------------------
std::vector<vk::PhysicalDevice> GetPhysicalDevices(const vk::Instance& inst);

// -----------------------------------------------------------------------------
// ---------------------------------- Surface ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueSurfaceKHR CreateSurface(const vk::Instance& instance,
                                   GLFWwindow* window);

// -----------------------------------------------------------------------------
// -------------------------------- Queue Family -------------------------------
// -----------------------------------------------------------------------------
void PrintQueueFamilyProps(const vk::PhysicalDevice& physical_device);

std::vector<uint32_t> GetQueueFamilyIdxs(
        const vk::PhysicalDevice& physical_device,
        const vk::QueueFlags& queue_flags = vk::QueueFlagBits::eGraphics);

uint32_t GetGraphicPresentQueueFamilyIdx(
        const vk::PhysicalDevice& physical_device,
        const vk::SurfaceKHR& surface,
        const vk::QueueFlags& queue_flags = vk::QueueFlagBits::eGraphics);

// -----------------------------------------------------------------------------
// ----------------------------------- Device ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueDevice CreateDevice(uint32_t queue_family_idx,
                              const vk::PhysicalDevice& physical_device,
                              uint32_t n_queues = 1,
                              bool swapchain_support = true);

// -----------------------------------------------------------------------------
// ------------------------------- Command Buffer ------------------------------
// -----------------------------------------------------------------------------
struct CommandBuffersPack {
    vk::UniqueCommandPool pool;
    std::vector<vk::UniqueCommandBuffer> cmd_bufs;
};
CommandBuffersPack CreateCommandBuffers(
        const vk::Device& device, uint32_t queue_family_idx,
        uint32_t n_cmd_buffers = 1);

// -----------------------------------------------------------------------------
// --------------------------------- Swapchain ---------------------------------
// -----------------------------------------------------------------------------
struct SwapchainPack {
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::UniqueImageView> views;
    vk::Extent2D size;
};
SwapchainPack CreateSwapchain(const vk::PhysicalDevice& physical_device,
                              const vk::Device& device,
                              const vk::SurfaceKHR& surface, uint32_t win_w = 0,
                              uint32_t win_h = 0);

// -----------------------------------------------------------------------------
// ----------------------------------- Image -----------------------------------
// -----------------------------------------------------------------------------
struct ImagePack {
    vk::UniqueImage img;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory dev_mem;
};
ImagePack CreateImage(
        const vk::PhysicalDevice& physical_device, const vk::Device& device,
        const vk::Format format = vk::Format::eR8G8B8A8Uint,
        const vk::Extent2D& size = {256, 256},
        const vk::ImageUsageFlags& usage = vk::ImageUsageFlagBits::eSampled,
        const vk::MemoryPropertyFlags& memory_props =
                vk::MemoryPropertyFlagBits::eDeviceLocal,
        const vk::ImageAspectFlags& aspects = vk::ImageAspectFlagBits::eColor,
        bool is_staging = false, bool is_shared = false);

// -----------------------------------------------------------------------------
// ----------------------------------- Buffer ----------------------------------
// -----------------------------------------------------------------------------
struct BufferPack {
    vk::UniqueBuffer buf;
    vk::UniqueDeviceMemory dev_mem;
};
BufferPack CreateBuffer(const vk::PhysicalDevice& physical_device,
                        const vk::Device& device,
                        const vk::DeviceSize& size = 256,
                        const vk::BufferUsageFlags& usage =
                                vk::BufferUsageFlagBits::eVertexBuffer,
                        const vk::MemoryPropertyFlags& memory_props =
                                vk::MemoryPropertyFlagBits::eDeviceLocal);

}  // namespace vkw

#endif /* end of include guard */
