#include <vkw/vkw_context.h>

#include "vkw/vkw.h"
#include "vulkan/vulkan.hpp"

namespace vkw {

namespace {
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

const vk::Extent2D& SelectImgSize(const vk::Extent2D& size_org,
                                  const SwapchainPackPtr& swapchain_pack) {
    // Escape empty swapchain
    if (!swapchain_pack) {
        return size_org;
    }

    if (size_org.width == 0 || size_org.height == 0) {
        // Use swapchain size
        return swapchain_pack->size;
    } else {
        // Use valid original size
        return size_org;
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
}  // namespace

// -----------------------------------------------------------------------------
// ---------------------------------- Context ----------------------------------
// -----------------------------------------------------------------------------
void Context::init(const ContextInfo& ci, const WindowPtr& window) {
    // Set window pointer
    m_surface_enable = static_cast<bool>(window);
    m_window = window;

    // Create instance
    m_instance = CreateInstance(ci.app_name, ci.app_version, ci.engine_name,
                                ci.engine_version, ci.debug_enable,
                                m_surface_enable);

    // Get a physical_device
    m_physical_device = GetFirstPhysicalDevice(m_instance);

    // Create surface
    if (m_surface_enable) {
        m_surface = CreateSurface(m_instance, m_window);
        m_surface_format = GetSurfaceFormat(m_physical_device, m_surface);
    }

    // Select queue family index
    if (m_surface_enable) {
        m_queue_family_idx =
                GetGraphicPresentQueueFamilyIdx(m_physical_device, m_surface);
    } else {
        m_queue_family_idx =
                GetQueueFamilyIdx(m_physical_device, ci.queue_flags);
    }
    // Create device
    m_device = CreateDevice(m_queue_family_idx, m_physical_device, ci.n_queues,
                            m_surface_enable);

    if (m_surface_enable) {
        // Create swapchain
        m_swapchain_pack =
                CreateSwapchainPack(m_physical_device, m_device, m_surface);
    }

    // Get queues
    m_queues.reserve(ci.n_queues);
    for (uint32_t i = 0; i < ci.n_queues; i++) {
        m_queues.push_back(GetQueue(m_device, m_queue_family_idx, i));
    }
}

CommandBuffersPackPtr Context::createCmdBufsPack(uint32_t n_cmd_bufs) const {
    // Create command buffers
    return CreateCommandBuffersPack(m_device, m_queue_family_idx, n_cmd_bufs);
}

BufferPackPtr Context::createHostVisibBuffer(
        const vk::DeviceSize& size, const vk::BufferUsageFlags& usage) {
    // Create device host visible & coherent buffer
    return CreateBufferPack(m_physical_device, m_device, size, usage,
                            vk::MemoryPropertyFlagBits::eHostVisible |
                                    vk::MemoryPropertyFlagBits::eHostCoherent);
}

BufferPackPtr Context::createDeviceBuffer(const vk::DeviceSize& size,
                                          const vk::BufferUsageFlags& usage) {
    // Create device local buffer
    return CreateBufferPack(m_physical_device, m_device, size, usage,
                            vk::MemoryPropertyFlagBits::eDeviceLocal);
}

ImagePackPtr Context::createColorImagePack(const vk::Format& format,
                                           const vk::Extent2D& size_org,
                                           const vk::ImageUsageFlags& usage) {
    // Create image with color aspect
    auto size = SelectImgSize(size_org, m_swapchain_pack);
    return CreateImagePack(m_physical_device, m_device, format, size, usage);
}

ImagePackPtr Context::createDepthImagePack(const vk::Format& format,
                                           const vk::Extent2D& size_org) {
    // Create image with depth aspect
    auto size = SelectImgSize(size_org, m_swapchain_pack);
    return CreateImagePack(m_physical_device, m_device, format, size,
                           vk::ImageUsageFlagBits::eDepthStencilAttachment, {},
                           true, vk::ImageAspectFlagBits::eDepth);
}

TexturePackPtr Context::createTexturePack(const ImagePackPtr& img_pack,
                                          const vk::Filter& filter,
                                          const vk::SamplerAddressMode& addr) {
    // Create texture
    return CreateTexturePack(img_pack, m_device, filter, filter,
                             vk::SamplerMipmapMode::eLinear, addr, addr, addr);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

}  // namespace vkw
