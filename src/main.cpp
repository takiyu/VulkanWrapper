#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
    auto command_buffers_pack =
            vkw::CreateCommandBuffers(device, queue_family_idx, n_cmd_buffers);
    auto& command_buffer = command_buffers_pack.cmd_bufs[0];

    auto swapchain_pack = vkw::CreateSwapchain(physical_device, device,
                                               surface, win_w, win_h);

    const auto depth_format = vk::Format::eD16Unorm;
    auto depth_img_pack =
            vkw::CreateImage(physical_device, device, depth_format,
                             swapchain_pack.size,
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
        const glm::mat4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                                    0.0f, 0.0f, 0.5f, 1.0f};  // vulkan clip space
                                                              // has inverted y and
                                                              // half z !
        mvpc_mat = clip_mat * proj_mat * view_mat * model_mat;
    }

    auto uniform_buf_pack = vkw::CreateBuffer(
            physical_device, device, sizeof(mvpc_mat),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
    vkw::SendToDevice(device, uniform_buf_pack, &mvpc_mat[0], 16 * sizeof(float));

    auto desc_set_pack = vkw::CreateDescriptorSet(device, { {vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex} });
//     auto tex_pack = vkw::CreateTexture(vkw::CreateImage(physical_device, device), device);
//     auto desc_set_pack = vkw::CreateDescriptorSet(device, { {vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex},
//                                                             {vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eVertex} });

    vk::DescriptorBufferInfo desc_buf_info(uniform_buf_pack.buf.get(), 0, sizeof(glm::mat4x4));
    device->updateDescriptorSets(vk::WriteDescriptorSet(desc_set_pack.desc_set.get(), 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &desc_buf_info), {});

    vkw::WriteDescSetPack write_desc_set_pack;
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 0, uniform_buf_pack);
//     vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 1, tex_pack);

    vkw::UpdateDescriptorSets(device, write_desc_set_pack);

//
//     vk::UniquePipelineLayout pipeline_layout = device->createPipelineLayoutUnique({vk::PipelineLayoutCreateFlags(), 1, &descset_layout.get()});
//
//
//
//     vk::AttachmentDescription attach_descs[2];
//     attach_descs[0] = vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), surface_format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
//       vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);
//     attach_descs[1] = vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), depth_format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
//       vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
//
//     vk::AttachmentReference color_ref(0, vk::ImageLayout::eColorAttachmentOptimal);
//     vk::AttachmentReference depth_ref(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
//     vk::SubpassDescription subpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_ref, nullptr, &depth_ref);
//
//     vk::UniqueRenderPass render_pass = device->createRenderPassUnique({vk::RenderPassCreateFlags(), 2, attach_descs, 1, &subpass});




    // clang-format off
//
//     while (!glfwWindowShouldClose(window.get())) {
//         glfwPollEvents();
//         break;
//     }

    std::cout << "exit" << std::endl;

    return 0;
}
