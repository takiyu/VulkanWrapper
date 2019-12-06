#include <bits/stdint-uintn.h>

#include <cstdlib>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

#include "vkw.h"

int main(int argc, char const* argv[]) {
    (void)argc, (void)argv;

    const std::string app_name = "app name";
    const int app_version = 1;
    const std::string engine_name = "engine name";
    const int engine_version = 1;
    uint32_t win_w = 200;
    uint32_t win_h = 200;

    auto window = vkw::InitGLFWWindow(app_name, win_w, win_h);
    auto instance = vkw::CreateInstance(app_name, app_version, engine_name,
                                        engine_version);
    auto surface = vkw::CreateSurface(*instance, window.get());
    auto physical_device = vkw::GetPhysicalDevices(*instance).front();

    vkw::PrintQueueFamilyProps(physical_device);

    uint32_t queue_family_idx =
            vkw::GetGraphicPresentQueueFamilyIdx(physical_device, *surface);

    uint32_t n_queues = 1;
    auto device = vkw::CreateDevice(queue_family_idx, physical_device, n_queues,
                                    true);

    uint32_t n_cmd_buffers = 1;
    auto command_buffers_pack =
            vkw::CreateCommandBuffers(*device, queue_family_idx, n_cmd_buffers);
    auto& command_buffer = command_buffers_pack.cmd_bufs[0];

    auto swapchain_pack = vkw::CreateSwapchain(physical_device, *device,
                                               *surface, win_w, win_h);

    auto depth_img_pack =
            vkw::CreateImage(physical_device, *device, vk::Format::eD16Unorm,
                             swapchain_pack.size,
                             vk::ImageUsageFlagBits::eDepthStencilAttachment,
                             vk::MemoryPropertyFlagBits::eDeviceLocal,
                             vk::ImageAspectFlagBits::eDepth, true, false);

    const glm::mat4 model_mat = glm::mat4(1.0f);
    const glm::mat4 view_mat = glm::lookAt(glm::vec3(-5.0f, 3.0f, -10.0f),
                                           glm::vec3(0.0f, 0.0f, 0.0f),
                                           glm::vec3(0.0f, -1.0f, 0.0f));
    const glm::mat4 proj_mat =
            glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    const glm::mat4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                                0.0f, 0.0f, 0.5f, 1.0f};  // vulkan clip space
                                                          // has inverted y and
                                                          // half z !
    const glm::mat4 mvpc_mat = clip_mat * proj_mat * view_mat * model_mat;

    auto uniform_buf_pack = vkw::CreateBuffer(
            physical_device, *device, sizeof(mvpc_mat),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);

    // clang-format off
//
//     while (!glfwWindowShouldClose(window.get())) {
//         glfwPollEvents();
//         break;
//     }

    std::cout << "exit" << std::endl;

    return 0;
}
