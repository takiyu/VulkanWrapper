#ifndef VKW_H_20191226
#define VKW_H_20191226

#include <vkw/vkw_base.h>

namespace vkw {

class GraphicsContext;
using GraphicsContextPtr = std::shared_ptr<GraphicsContext>;
class Image;
using ImagePtr = std::shared_ptr<Image>;
class Buffer;
using BufferPtr = std::shared_ptr<Buffer>;

// -----------------------------------------------------------------------------
// ------------------------------ Graphics Context -----------------------------
// -----------------------------------------------------------------------------
class GraphicsContext : public std::enable_shared_from_this<GraphicsContext> {
public:
    template <typename... Args>
    static auto Create(const Args&... args) {
        return GraphicsContextPtr(new GraphicsContext(args...));
    }

    const WindowPtr& getWindow() const;
    const vk::UniqueInstance& getInstance() const;
    const vk::PhysicalDevice& getPhysicalDevice() const;
    const vk::UniqueDevice& getDevice() const;
    const vk::UniqueSurfaceKHR& getSurface() const;
    const SwapchainPackPtr& getSwapchainPack() const;
    const std::vector<vk::Queue>& getQueues() const;

    uint32_t getQueueFamilyIdx() const;
    vk::Format getSurfaceFormat() const;

    ImagePtr createImage(
            const vk::Format& format = vk::Format::eR8G8B8A8Uint,
            const vk::Extent2D& size = {256, 256},
            const vk::ImageUsageFlags& usage = vk::ImageUsageFlagBits::eSampled,
            const vk::MemoryPropertyFlags& memory_props =
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
            const vk::ImageAspectFlags& aspects =
                    vk::ImageAspectFlagBits::eColor,
            bool is_staging = false, bool is_shared = false);

    BufferPtr createBuffer(const vk::DeviceSize& size = 256,
                           const vk::BufferUsageFlags& usage =
                                   vk::BufferUsageFlagBits::eVertexBuffer,
                           const vk::MemoryPropertyFlags& memory_props =
                                   vk::MemoryPropertyFlagBits::eDeviceLocal);

private:
    // Initialize without surface
    GraphicsContext(const std::string& app_name, uint32_t app_version,
                    uint32_t n_queues, bool debug_enable = true);
    // Initialize with surface
    GraphicsContext(const std::string& app_name, uint32_t app_version,
                    uint32_t n_queues, const vkw::WindowPtr& window,
                    bool debug_enable = true);

    vkw::WindowPtr m_window;
    vk::UniqueInstance m_instance;
    vk::PhysicalDevice m_physical_device;
    vk::UniqueDevice m_device;
    vk::UniqueSurfaceKHR m_surface;
    vkw::SwapchainPackPtr m_swapchain_pack;
    std::vector<vk::Queue> m_queues;

    uint32_t m_queue_family_idx;
    vk::Format m_surface_format;
};

// -----------------------------------------------------------------------------
// ----------------------------------- Image -----------------------------------
// -----------------------------------------------------------------------------
class Image {
public:
    friend class GraphicsContext;

    const vkw::ImagePackPtr& getImagePack() const;

    void sendToDevice(const void* data, uint64_t n_bytes);

    template <typename T>
    void sendToDevice(const std::vector<T>& vec) {
        sendToDevice(vec.data(), vec.size() * sizeof(T));
    }

    template <typename T>
    void sendToDevice(const T& v) {
        sendToDevice(&v, sizeof(T));
    }

private:
    Image();

    vkw::GraphicsContextPtr m_context;
    vkw::ImagePackPtr m_img_pack;
};

// -----------------------------------------------------------------------------
// ----------------------------------- Buffer ----------------------------------
// -----------------------------------------------------------------------------
class Buffer {
public:
    friend class GraphicsContext;

    const vkw::BufferPackPtr& getBufferPack() const;

    void sendToDevice(const void* data, uint64_t n_bytes);

    template <typename T>
    void sendToDevice(const std::vector<T>& vec) {
        sendToDevice(vec.data(), vec.size() * sizeof(T));
    }

    template <typename T>
    void sendToDevice(const T& v) {
        sendToDevice(&v, sizeof(T));
    }

private:
    Buffer();

    vkw::GraphicsContextPtr m_context;
    vkw::BufferPackPtr m_buf_pack;
};

}  // namespace vkw

#endif  // end of include guard
