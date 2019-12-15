#ifndef VKW_H_20191130
#define VKW_H_20191130

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

namespace vkw {

// -----------------------------------------------------------------------------
// ----------------------------------- GLFW ------------------------------------
// -----------------------------------------------------------------------------
struct GLFWWindowDeleter {
    void operator()(GLFWwindow* ptr);
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
                                  uint32_t engine_version,
                                  bool debug_enable = true);

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
using CommandBuffersPackPtr = std::shared_ptr<CommandBuffersPack>;
CommandBuffersPackPtr CreateCommandBuffersPack(const vk::UniqueDevice& device,
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
using SwapchainPackPtr = std::shared_ptr<SwapchainPack>;
SwapchainPackPtr CreateSwapchainPack(
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
    vk::Extent2D size;
    vk::UniqueDeviceMemory dev_mem;
    vk::DeviceSize dev_mem_size;
};
using ImagePackPtr = std::shared_ptr<ImagePack>;
ImagePackPtr CreateImagePack(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueDevice& device,
        const vk::Format& format = vk::Format::eR8G8B8A8Uint,
        const vk::Extent2D& size = {256, 256},
        const vk::ImageUsageFlags& usage = vk::ImageUsageFlagBits::eSampled,
        const vk::MemoryPropertyFlags& memory_props =
                vk::MemoryPropertyFlagBits::eDeviceLocal,
        const vk::ImageAspectFlags& aspects = vk::ImageAspectFlagBits::eColor,
        bool is_staging = false, bool is_shared = false);

void SendToDevice(const vk::UniqueDevice& device, const ImagePackPtr& img_pack,
                  const void* data, uint64_t n_bytes);

struct TexturePack {
    ImagePackPtr img_pack;
    vk::UniqueSampler sampler;
};
using TexturePackPtr = std::shared_ptr<TexturePack>;
TexturePackPtr CreateTexturePack(
        const ImagePackPtr& img_pack, const vk::UniqueDevice& device,
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
    vk::DeviceSize size;
    vk::UniqueDeviceMemory dev_mem;
    vk::DeviceSize dev_mem_size;
};
using BufferPackPtr = std::shared_ptr<BufferPack>;
BufferPackPtr CreateBufferPack(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueDevice& device, const vk::DeviceSize& size = 256,
        const vk::BufferUsageFlags& usage =
                vk::BufferUsageFlagBits::eVertexBuffer,
        const vk::MemoryPropertyFlags& memory_props =
                vk::MemoryPropertyFlagBits::eDeviceLocal);

void SendToDevice(const vk::UniqueDevice& device, const BufferPackPtr& buf_pack,
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
using DescSetPackPtr = std::shared_ptr<DescSetPack>;
DescSetPackPtr CreateDescriptorSetPack(const vk::UniqueDevice& device,
                                       const std::vector<DescSetInfo>& info);

struct WriteDescSetPack {
    std::vector<vk::WriteDescriptorSet> write_desc_sets;
    std::vector<std::vector<vk::DescriptorImageInfo>> desc_img_info_vecs;
    std::vector<std::vector<vk::DescriptorBufferInfo>> desc_buf_info_vecs;
};
using WriteDescSetPackPtr = std::shared_ptr<WriteDescSetPack>;
WriteDescSetPackPtr CreateWriteDescSetPack();

void AddWriteDescSet(WriteDescSetPackPtr& write_pack,
                     const DescSetPackPtr& desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<TexturePackPtr>& tex_packs);
void AddWriteDescSet(WriteDescSetPackPtr& write_pack,
                     const DescSetPackPtr& desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<BufferPackPtr>& buf_packs);

void UpdateDescriptorSets(const vk::UniqueDevice& device,
                          const WriteDescSetPackPtr& write_desc_set_pack);

// -----------------------------------------------------------------------------
// -------------------------------- RenderPass ---------------------------------
// -----------------------------------------------------------------------------
struct RenderPassPack {
    std::vector<vk::AttachmentDescription> attachment_descs;
    std::vector<std::vector<vk::AttachmentReference>> attachment_ref_vecs;
    std::vector<vk::SubpassDescription> subpass_descs;
    vk::UniqueRenderPass render_pass;
    // TODO: dependency
};

using RenderPassPackPtr = std::shared_ptr<RenderPassPack>;
RenderPassPackPtr CreateRenderPassPack();
void AddAttachientDesc(
        RenderPassPackPtr& render_pass_pack,
        const vk::Format& format = vk::Format::eB8G8R8A8Unorm,
        const vk::AttachmentLoadOp& load_op = vk::AttachmentLoadOp::eClear,
        const vk::AttachmentStoreOp& store_op = vk::AttachmentStoreOp::eStore,
        const vk::ImageLayout& final_layout = vk::ImageLayout::ePresentSrcKHR);

using AttachmentIdx = uint32_t;
using AttachmentRefInfo = std::tuple<AttachmentIdx, vk::ImageLayout>;
void AddSubpassDesc(RenderPassPackPtr& render_pass_pack,
                    const std::vector<AttachmentRefInfo>& inp_attach_refs,
                    const std::vector<AttachmentRefInfo>& col_attach_refs,
                    const AttachmentRefInfo& depth_stencil_attach_ref = {
                            uint32_t(~0), vk::ImageLayout::eUndefined});

void UpdateRenderPass(const vk::UniqueDevice& device,
                      RenderPassPackPtr& render_pass_pack);

// -----------------------------------------------------------------------------
// -------------------------------- FrameBuffer --------------------------------
// -----------------------------------------------------------------------------
vk::UniqueFramebuffer CreateFrameBuffer(
        const vk::UniqueDevice& device,
        const RenderPassPackPtr& render_pass_pack,
        const std::vector<ImagePackPtr>& imgs,
        const vk::Extent2D& size = {0, 0});

std::vector<vk::UniqueFramebuffer> CreateFrameBuffers(
        const vk::UniqueDevice& device,
        const RenderPassPackPtr& render_pass_pack,
        const std::vector<ImagePackPtr>& imgs,
        const uint32_t swapchain_attach_idx, const SwapchainPackPtr& swapchain);

// -----------------------------------------------------------------------------
// -------------------------------- ShaderModule -------------------------------
// -----------------------------------------------------------------------------
struct ShaderModulePack {
    vk::UniqueShaderModule shader_module;
    vk::ShaderStageFlagBits stage;
};
using ShaderModulePackPtr = std::shared_ptr<ShaderModulePack>;

class GLSLCompiler {
public:
    GLSLCompiler();
    ~GLSLCompiler();
    ShaderModulePackPtr compileFromString(
            const vk::UniqueDevice& device, const std::string& source,
            const vk::ShaderStageFlagBits& stage =
                    vk::ShaderStageFlagBits::eVertex);
};

// -----------------------------------------------------------------------------
// ---------------------------------- Pipeline ---------------------------------
// -----------------------------------------------------------------------------
struct VtxInputBindingInfo {
    uint32_t binding_idx = 0;
    uint32_t stride = sizeof(float);
};
struct VtxInputAttribInfo {
    uint32_t location = 0;     // Location in shader
    uint32_t binding_idx = 0;  // Select a binding
    vk::Format format = vk::Format::eR32G32B32A32Sfloat;
    uint32_t offset = 0;  // Offset in the binding buffer
};

struct PipelinePack {
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
};
using PipelinePackPtr = std::shared_ptr<PipelinePack>;
PipelinePackPtr CreatePipeline(
        const vk::UniqueDevice& device,
        const std::vector<ShaderModulePackPtr>& shader_modules,
        const std::vector<VtxInputBindingInfo>& vtx_inp_binding_info,
        const std::vector<VtxInputAttribInfo>& vtx_inp_attrib_info,
        const DescSetPackPtr& desc_set_pack,
        const RenderPassPackPtr& render_pass_pack);

}  // namespace vkw

#endif /* end of include guard */
