#ifndef VKW_H_20191130
#define VKW_H_20191130

#include <bits/stdint-uintn.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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
using UniqueGLFWWindow = std::unique_ptr<GLFWwindow, GLFWWindowDeleter>;

UniqueGLFWWindow InitGLFWWindow(const std::string& win_name, uint32_t win_w,
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
std::vector<vk::PhysicalDevice> GetPhysicalDevices(
        const vk::UniqueInstance& instance);

// -----------------------------------------------------------------------------
// ---------------------------------- Surface ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueSurfaceKHR CreateSurface(const vk::UniqueInstance& instance,
                                   const UniqueGLFWWindow& window);

vk::Format GetSurfaceFormat(const vk::PhysicalDevice& physical_device,
                            const vk::UniqueSurfaceKHR& surface);

// -----------------------------------------------------------------------------
// -------------------------------- Queue Family -------------------------------
// -----------------------------------------------------------------------------
void PrintQueueFamilyProps(const vk::PhysicalDevice& physical_device);

std::vector<uint32_t> GetQueueFamilyIdxs(
        const vk::PhysicalDevice& physical_device,
        const vk::QueueFlags& queue_flags = vk::QueueFlagBits::eGraphics);

uint32_t GetGraphicPresentQueueFamilyIdx(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueSurfaceKHR& surface,
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
CommandBuffersPack CreateCommandBuffers(const vk::UniqueDevice& device,
                                        uint32_t queue_family_idx,
                                        uint32_t n_cmd_buffers = 1);

// -----------------------------------------------------------------------------
// --------------------------------- Swapchain ---------------------------------
// -----------------------------------------------------------------------------
struct SwapchainPack {
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::UniqueImageView> views;
    vk::Extent2D size;
};
SwapchainPack CreateSwapchain(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueDevice& device, const vk::UniqueSurfaceKHR& surface,
        uint32_t win_w = 0, uint32_t win_h = 0,
        const vk::Format& surface_format = vk::Format::eUndefined,
        const vk::ImageUsageFlags& usage =
                vk::ImageUsageFlagBits::eColorAttachment);

// -----------------------------------------------------------------------------
// ----------------------------------- Image -----------------------------------
// -----------------------------------------------------------------------------
struct ImagePack {
    vk::UniqueImage img;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory dev_mem;
    vk::DeviceSize dev_mem_size;
};
ImagePack CreateImage(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueDevice& device,
        const vk::Format& format = vk::Format::eR8G8B8A8Uint,
        const vk::Extent2D& size = {256, 256},
        const vk::ImageUsageFlags& usage = vk::ImageUsageFlagBits::eSampled,
        const vk::MemoryPropertyFlags& memory_props =
                vk::MemoryPropertyFlagBits::eDeviceLocal,
        const vk::ImageAspectFlags& aspects = vk::ImageAspectFlagBits::eColor,
        bool is_staging = false, bool is_shared = false);

void SendToDevice(const vk::UniqueDevice& device, const ImagePack& img_pack,
                  const void* data, uint64_t n_bytes);

struct TexturePack {
    ImagePack img_pack;
    vk::UniqueSampler sampler;
};
TexturePack CreateTexture(
        ImagePack&& img_pack, const vk::UniqueDevice& device,
        const vk::Filter& mag_filter = vk::Filter::eLinear,
        const vk::Filter& min_filter = vk::Filter::eLinear,
        const vk::SamplerMipmapMode& mipmap = vk::SamplerMipmapMode::eLinear,
        const vk::SamplerAddressMode& addr_u = vk::SamplerAddressMode::eRepeat,
        const vk::SamplerAddressMode& addr_v = vk::SamplerAddressMode::eRepeat,
        const vk::SamplerAddressMode& addr_w = vk::SamplerAddressMode::eRepeat);

// -----------------------------------------------------------------------------
// ----------------------------------- Buffer ----------------------------------
// -----------------------------------------------------------------------------
struct BufferPack {
    vk::UniqueBuffer buf;
    vk::UniqueDeviceMemory dev_mem;
    vk::DeviceSize dev_mem_size;
};
BufferPack CreateBuffer(const vk::PhysicalDevice& physical_device,
                        const vk::UniqueDevice& device,
                        const vk::DeviceSize& size = 256,
                        const vk::BufferUsageFlags& usage =
                                vk::BufferUsageFlagBits::eVertexBuffer,
                        const vk::MemoryPropertyFlags& memory_props =
                                vk::MemoryPropertyFlagBits::eDeviceLocal);

void SendToDevice(const vk::UniqueDevice& device, const BufferPack& buf_pack,
                  const void* data, uint64_t n_bytes);

// -----------------------------------------------------------------------------
// ------------------------------- DescriptorSet -------------------------------
// -----------------------------------------------------------------------------
using DescCnt = uint32_t;
using DescSetInfo =
        std::tuple<vk::DescriptorType, DescCnt, vk::ShaderStageFlags>;
struct DescSetPack {
    vk::UniqueDescriptorSetLayout desc_set_layout;
    vk::UniqueDescriptorPool desc_pool;
    vk::UniqueDescriptorSet desc_set;
    std::vector<DescSetInfo> desc_set_info;
};
DescSetPack CreateDescriptorSet(const vk::UniqueDevice& device,
                                const std::vector<DescSetInfo>& info);

struct WriteDescSetPack {
    std::vector<vk::WriteDescriptorSet> write_desc_sets;
    std::vector<std::vector<vk::DescriptorImageInfo>> desc_img_infos;
    std::vector<std::vector<vk::DescriptorBufferInfo>> desc_buf_infos;
};
void AddWriteDescSet(WriteDescSetPack& write_pack,
                     const DescSetPack& desc_set_pack,
                     const uint32_t binding_idx, const TexturePack& tex_pack);
void AddWriteDescSet(WriteDescSetPack& write_pack,
                     const DescSetPack& desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<TexturePack>& tex_packs);
void AddWriteDescSet(WriteDescSetPack& write_pack,
                     const DescSetPack& desc_set_pack,
                     const uint32_t binding_idx, const BufferPack& buf_pack);
void AddWriteDescSet(WriteDescSetPack& write_pack,
                     const DescSetPack& desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<BufferPack>& buf_packs);

void UpdateDescriptorSets(const vk::UniqueDevice& device,
                          const WriteDescSetPack& write_desc_set_pack);

}  // namespace vkw

#endif /* end of include guard */
