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
#include <sstream>

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void RunExampleApp11(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    (void)window;
    (void)draw_hook;

    // Initialize without display environment
    const bool display_enable = false;
    const bool debug_enable = true;
    const uint32_t n_queues = 1;

    // Create instance
    auto instance = vkw::CreateInstance("VKW Example 11", 1, "VKW", 0,
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
    const uint32_t IMG_W = 256;
    const uint32_t IMG_H = 256;
    const uint32_t IMG_C = 4;
    const uint32_t BUF_SIZE = IMG_W * IMG_H * IMG_C;

    // Create buffer & image
    auto buf_img_pack = vkw::CreateBufferImagePack(
            physical_device, device, true, BUF_SIZE, vk::Format::eR8G8B8A8Uint,
            {IMG_W, IMG_H}, 1,
            vk::BufferUsageFlagBits::eTransferSrc |
                    vk::BufferUsageFlagBits::eTransferDst,
            vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eTransferSrc |
                    vk::ImageUsageFlagBits::eTransferDst);
    auto buf = std::get<0>(buf_img_pack);
    auto img = std::get<1>(buf_img_pack);

    // Create intermediate buffer
    auto buf_mid1 =
            vkw::CreateBufferPack(physical_device, device, BUF_SIZE,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                          vk::BufferUsageFlagBits::eTransferDst,
                                  vkw::HOST_VISIB_COHER_PROPS);
    auto buf_mid2 =
            vkw::CreateBufferPack(physical_device, device, BUF_SIZE,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                          vk::BufferUsageFlagBits::eTransferDst,
                                  vkw::HOST_VISIB_COHER_PROPS);

    // Create original CPU data
    std::vector<uint8_t> org_data(BUF_SIZE);
    for (uint32_t i = 0; i < BUF_SIZE; i++) {
        org_data[i] = static_cast<uint8_t>(i);
    }

    // Send from CPU to intermediate buffer
    vkw::SendToDevice(device, buf_mid1, org_data.data(), BUF_SIZE);

    // Copy from intermediate buffer to image
    vkw::BeginCommand(cmd_buf);
    vkw::CopyBufferToImage(cmd_buf, buf_mid1, img);
    vkw::EndCommand(cmd_buf);
    // Execute
    auto fence1 = vkw::CreateFence(device);
    vkw::QueueSubmit(queues[0], cmd_buf, fence1);
    vkw::WaitForFences(device, {fence1});

    // Copy from buffer to intermediate buffer
    vkw::BeginCommand(cmd_buf);
    vkw::CopyBuffer(cmd_buf, buf, buf_mid2);
    vkw::EndCommand(cmd_buf);
    // Execute
    auto fence2 = vkw::CreateFence(device);
    vkw::QueueSubmit(queues[0], cmd_buf, fence2);
    vkw::WaitForFences(device, {fence2});

    // Receive from destination buffer
    std::vector<uint8_t> res_data(BUF_SIZE);
    vkw::RecvFromDevice(device, buf_mid2, res_data.data(), BUF_SIZE);
    bool is_all_same = true;
    for (uint32_t i = 0; i < BUF_SIZE; i++) {
        if (res_data[i] != org_data[i]) {
            is_all_same = false;
            std::stringstream ss;
            ss << "Received value is wrong. ("
               << static_cast<uint32_t>(res_data[i]) << " vs "
               << static_cast<uint32_t>(org_data[i]) << ")";
            vkw::PrintErr(ss.str());
        }
    }
    if (is_all_same) {
        vkw::PrintInfo("Success of bi-directional image transfer.");
    }
}
