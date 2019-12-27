#ifndef VKW_H_20191226
#define VKW_H_20191226

#include <vkw/vkw_base.h>

namespace vkw {

// -----------------------------------------------------------------------------
// ------------------------------ Graphics Context -----------------------------
// -----------------------------------------------------------------------------
class GraphicsContext;
using GraphicsContextPtr = std::shared_ptr<GraphicsContext>;
class GraphicsContext {
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
class Image;
using ImagePtr = std::shared_ptr<Image>;
class Image {
public:
private:
};

}  // namespace vkw

#endif  // end of include guard
