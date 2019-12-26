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
    static GraphicsContextPtr Create();

    // Initialize without surface
    void init(const std::string& app_name, uint32_t app_version,
              uint32_t n_queues, bool debug_enable = true);
#if defined(__ANDROID__)
    // Initialize with Android surface
    void init(const std::string& app_name, uint32_t app_version, JNIEnv* jenv,
              jobject jsurface, uint32_t n_queues, bool debug_enable = true);
#else
    // Initialize with GLFW surface
    void init(const std::string& app_name, uint32_t app_version, uint32_t win_w,
              uint32_t win_h, uint32_t n_queues, bool debug_enable = true);
#endif

#if defined(__ANDROID__)
    const UniqueANativeWindow& getANativeWindow() const;
#else
    const UniqueGLFWWindow& getGLFWWindow() const;
#endif
    const vk::UniqueInstance& getInstance() const;
    const vk::PhysicalDevice& getPhysicalDevice() const;
    const vk::UniqueDevice& getDevice() const;
    const vk::UniqueSurfaceKHR& getSurface() const;
    const SwapchainPackPtr& getSwapchainPack() const;
    const std::vector<vk::Queue>& getQueues() const;

    uint32_t getQueueFamilyIdx() const;
    vk::Format getSurfaceFormat() const;

private:
    GraphicsContext();

#if defined(__ANDROID__)
    UniqueANativeWindow m_anative_window;
#else
    UniqueGLFWWindow m_glfw_window;
#endif
    vk::UniqueInstance m_instance;
    vk::PhysicalDevice m_physical_device;
    vk::UniqueDevice m_device;
    vk::UniqueSurfaceKHR m_surface;
    SwapchainPackPtr m_swapchain_pack;
    std::vector<vk::Queue> m_queues;

    uint32_t m_queue_family_idx;
    vk::Format m_surface_format;
};

}  // namespace vkw

#endif  // end of include guard
