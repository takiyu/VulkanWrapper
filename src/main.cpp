#include <cstdlib>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "vkw.h"

int main(int argc, char const *argv[]) {
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
    auto surface = vkw::CreateSurface(instance, window);
    auto physical_device = vkw::GetPhysicalDevices(instance).front();
    const auto surface_format = vkw::GetSurfaceFormat(physical_device, surface);

    vkw::PrintQueueFamilyProps(physical_device);

    uint32_t queue_family_idx =
            vkw::GetGraphicPresentQueueFamilyIdx(physical_device, surface);

    uint32_t n_queues = 1;
    auto device = vkw::CreateDevice(queue_family_idx, physical_device, n_queues,
                                    true);

    uint32_t n_cmd_buffers = 1;
    auto command_buffers_pack = vkw::CreateCommandBuffersPack(
            device, queue_family_idx, n_cmd_buffers);
    auto &command_buffer = command_buffers_pack->cmd_bufs[0];

    auto swapchain_pack = vkw::CreateSwapchainPack(physical_device, device,
                                                   surface, win_w, win_h);

    const auto depth_format = vk::Format::eD16Unorm;
    auto depth_img_pack = vkw::CreateImagePack(
            physical_device, device, depth_format, swapchain_pack->size,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::ImageAspectFlagBits::eDepth, true, false);

    glm::mat4 mvpc_mat;
    {
        const glm::mat4 model_mat = glm::mat4(1.0f);
        const glm::mat4 view_mat = glm::lookAt(glm::vec3(-5.0f, 3.0f, -10.0f),
                                               glm::vec3(0.0f, 0.0f, 0.0f),
                                               glm::vec3(0.0f, -1.0f, 0.0f));
        const glm::mat4 proj_mat =
                glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
        // vulkan clip space has inverted y and half z !
        const glm::mat4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                                    0.0f, 0.0f, 0.5f, 1.0f};
        mvpc_mat = clip_mat * proj_mat * view_mat * model_mat;
    }

    auto uniform_buf_pack = vkw::CreateBufferPack(
            physical_device, device, sizeof(mvpc_mat),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
    vkw::SendToDevice(device, uniform_buf_pack, &mvpc_mat[0],
                      16 * sizeof(float));

#if 1
    auto desc_set_pack = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eUniformBuffer, 1,
                      vk::ShaderStageFlagBits::eVertex}});
#else
    auto tex_pack = vkw::CreateTexture(
            vkw::CreateImage(physical_device, device), device);
    auto desc_set_pack = vkw::CreateDescriptorSet(
            device, {{vk::DescriptorType::eUniformBuffer, 1,
                      vk::ShaderStageFlagBits::eVertex},
                     {vk::DescriptorType::eCombinedImageSampler, 1,
                      vk::ShaderStageFlagBits::eVertex}});
#endif

    auto write_desc_set_pack = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 0,
                         {uniform_buf_pack});
#if 0
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 1, {tex_pack});
#endif
    vkw::UpdateDescriptorSets(device, write_desc_set_pack);

    auto render_pass_pack = vkw::CreateRenderPassPack();
    vkw::AddAttachientDesc(
            render_pass_pack, surface_format, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::ePresentSrcKHR);
    vkw::AddAttachientDesc(render_pass_pack, depth_format,
                           vk::AttachmentLoadOp::eClear,
                           vk::AttachmentStoreOp::eDontCare,
                           vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vkw::AddSubpassDesc(render_pass_pack, {},
                        {
                                {0, vk::ImageLayout::eColorAttachmentOptimal},
                        },
                        {1, vk::ImageLayout::eDepthStencilAttachmentOptimal});
    vkw::UpdateRenderPass(device, render_pass_pack);

    auto frame_buffers = vkw::CreateFrameBuffers(device, render_pass_pack, {nullptr, depth_img_pack}, 0, swapchain_pack);

    // clang-format off

//
//     vk::UniquePipelineLayout pipeline_layout = device->createPipelineLayoutUnique({vk::PipelineLayoutCreateFlags(), 1, &descset_layout.get()});


//
//     while (!glfwWindowShouldClose(window.get())) {
//         glfwPollEvents();
//         break;
//     }

    std::cout << "exit" << std::endl;

    return 0;
}
