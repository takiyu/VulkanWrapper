#include "app.h"

#include <vkw/warning_suppressor.h>
#include "vkw/vkw.h"

BEGIN_VKW_SUPPRESS_WARNING
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
END_VKW_SUPPRESS_WARNING

#include <iostream>

// -----------------------------------------------------------------------------
void RunExampleApp07(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    (void)window;
    (void)draw_hook;

    // Initialize without display environment
    const bool display_enable = false;
    const bool debug_enable = true;
    const uint32_t n_queues = 1;

    // Create instance
    auto instance = vkw::CreateInstance("VKW Example 07", 1, "VKW", 0,
                                        debug_enable, display_enable);
    // Get a physical_device
    auto physical_device = vkw::GetFirstPhysicalDevice(instance);

    // Select queue family
    uint32_t queue_family_idx = vkw::GetQueueFamilyIdxs(physical_device)[0];
    // Create device
    auto device = vkw::CreateDevice(queue_family_idx, physical_device, n_queues,
                                    display_enable);

    // Get queues
    std::vector<vk::Queue> queues;
    queues.reserve(n_queues);
    for (uint32_t i = 0; i < n_queues; i++) {
        queues.push_back(vkw::GetQueue(device, queue_family_idx, i));
    }

    // Get command buffer
    const uint32_t n_cmd_bufs = 1;
    auto cmd_bufs_pack =
            vkw::CreateCommandBuffersPack(device, queue_family_idx, n_cmd_bufs);
    auto& cmd_buf = cmd_bufs_pack->cmd_bufs[0];

    // ------------------
    const uint32_t DATA_SIZE = 10 * 10 * 4;

    // Create source buffer
    auto buf_src = vkw::CreateBufferPack(physical_device, device, DATA_SIZE,
                vk::BufferUsageFlagBits::eTransferSrc, vkw::HOST_VISIB_COHER_PROPS);

    // Create target image
    auto img = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR8G8B8A8Uint, {10, 10},
            vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eColorAttachment |
                    vk::ImageUsageFlagBits::eTransferDst |
                    vk::ImageUsageFlagBits::eTransferSrc,
            {}, true);

    // Create destination buffer
    auto buf_dst = vkw::CreateBufferPack(physical_device, device, DATA_SIZE,
                vk::BufferUsageFlagBits::eTransferDst, vkw::HOST_VISIB_COHER_PROPS);

    // Create original CPU data
    std::vector<uint8_t> org_data(DATA_SIZE);
    for (uint32_t i = 0; i < DATA_SIZE; i++) {
        org_data[i] = static_cast<uint8_t>(i);
    }

    // Send to source buffer
    vkw::SendToDevice(device, buf_src, org_data.data(), DATA_SIZE);

    // Copy from buffer to image
    vkw::BeginCommand(cmd_buf);
    vkw::CopyBufferToImage(cmd_buf, buf_src, img);
    vkw::EndCommand(cmd_buf);
    // Execute
    auto fence = vkw::CreateFence(device);
    vkw::QueueSubmit(queues[0], cmd_buf, fence);
    vkw::WaitForFences(device, {fence});

    // Copy from image to buffer
    vkw::BeginCommand(cmd_buf);
    vkw::CopyImageToBuffer(cmd_buf, img, buf_dst);
    vkw::EndCommand(cmd_buf);
    // Execute
    vkw::ResetFence(device, fence);
    vkw::QueueSubmit(queues[0], cmd_buf, fence);
    vkw::WaitForFences(device, {fence});

    // Receive from destination buffer
    std::vector<uint8_t> res_data(DATA_SIZE);
    vkw::RecvFromDevice(device, buf_dst, res_data.data(), DATA_SIZE);
    bool is_all_same = true;
    for (uint32_t i = 0; i < DATA_SIZE; i++) {
        if (res_data[i] != org_data[i]) {
            is_all_same = false;
            vkw::PrintErr("Received value is wrong.");
        }
    }
    if (is_all_same) {
        vkw::PrintInfo("Success of bi-directional image transfer.");
    }
}
