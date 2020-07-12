#ifndef VKW_CONTEXT_H_20200625
#define VKW_CONTEXT_H_20200625

#include <vkw/vkw.h>

namespace vkw {

// -----------------------------------------------------------------------------
// ------------------------------- Context Info --------------------------------
// -----------------------------------------------------------------------------
struct ContextInfo {
    // Instance
    std::string app_name = "app";
    uint32_t app_version = 0;
    std::string engine_name = "VKW";
    uint32_t engine_version = 0;
    bool debug_enable = true;
    // Queue
    vk::QueueFlags queue_flags = vk::QueueFlagBits::eGraphics |
                                 vk::QueueFlagBits::eCompute |
                                 vk::QueueFlagBits::eTransfer;
    uint32_t n_queues = 1;
};

// -----------------------------------------------------------------------------
// ---------------------------------- Context ----------------------------------
// -----------------------------------------------------------------------------
class Context;
using ContextPtr = std::shared_ptr<Context>;
class Context {
public:
    // -------------------------------------------------------------------------
    // ---------------------------- Creator Method -----------------------------
    // -------------------------------------------------------------------------
    template <typename... T>
    static auto Create(T&&... args) {
        return ContextPtr(new Context(std::forward<T>(args)...));
    }

    // -------------------------------------------------------------------------
    // -------------------------------- Methods --------------------------------
    // -------------------------------------------------------------------------
    void init(const ContextInfo& ci, const WindowPtr& window = nullptr);

    CommandBuffersPackPtr createCmdBufsPack(uint32_t n_cmd_bufs) const;

    BufferPackPtr createHostVisibBuffer(
            const vk::DeviceSize& size = 256,
            const vk::BufferUsageFlags& usage =
                    vk::BufferUsageFlagBits::eVertexBuffer);
    BufferPackPtr createDeviceBuffer(
            const vk::DeviceSize& size = 256,
            const vk::BufferUsageFlags& usage =
                    vk::BufferUsageFlagBits::eStorageBuffer);

    ImagePackPtr createColorImagePack(
            const vk::Format& format = vk::Format::eR8G8B8A8Uint,
            const vk::Extent2D& size = {0, 0},
            const vk::ImageUsageFlags& usage =
                    vk::ImageUsageFlagBits::eInputAttachment |
                    vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eColorAttachment);
    ImagePackPtr createDepthImagePack(
            const vk::Format& format = vk::Format::eD16Unorm,
            const vk::Extent2D& size = {0, 0});
    TexturePackPtr createTexturePack(
            const ImagePackPtr& img_pack,
            const vk::Filter& filetr = vk::Filter::eLinear,
            const vk::SamplerAddressMode& addr =
                    vk::SamplerAddressMode::eRepeat);

    // -------------------------------------------------------------------------
    // ------------------------ Public Member Variables ------------------------
    // -------------------------------------------------------------------------
    // Basic variables
    vk::UniqueInstance m_instance;
    vk::PhysicalDevice m_physical_device;
    uint32_t m_queue_family_idx;
    vk::UniqueDevice m_device;
    std::vector<vk::Queue> m_queues;
    GLSLCompiler glsl_compiler;

    // Surface variables
    bool m_surface_enable = false;
    WindowPtr m_window;
    vk::UniqueSurfaceKHR m_surface;
    vk::Format m_surface_format;
    SwapchainPackPtr m_swapchain_pack;

    // -------------------------------------------------------------------------
    // ----------------------------- Constructors ------------------------------
    // -------------------------------------------------------------------------
private:
    Context() {}

    template <typename... T>
    Context(T&&... args) {
        init(std::forward<T>(args)...);
    }

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
}  // namespace vkw

#endif  // end of include guard
