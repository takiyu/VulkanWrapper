#include <cstdlib>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "vkw.h"

// vertex shader with (P)osition and (C)olor in and (C)olor out
const std::string VERT_SOURCE = R"(
#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (std140, binding = 0) uniform buffer
{
    mat4 mvp;
} uniformBuffer;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;

layout (location = 0) out vec4 outColor;

void main()
{
    outColor = inColor;
    gl_Position = uniformBuffer.mvp * pos;
}
)";

// fragment shader with (C)olor in and (C)olor out
const std::string FRAG_SOURCE = R"(
#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec4 color;

layout (location = 0) out vec4 outColor;

void main()
{
    outColor = color;
}
)";

struct Vertex {
    float x, y, z, w;  // Position
    float r, g, b, a;  // Color
};
const std::vector<Vertex> CUBE_VERTICES = {
        // red face
        {-1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        // green face
        {-1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        // blue face
        {-1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        // yellow face
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        // magenta face
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        // cyan face
        {1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
};

int main(int argc, char const *argv[]) {
    (void)argc, (void)argv;

    const std::string app_name = "app name";
    const int app_version = 1;
    const std::string engine_name = "engine name";
    const int engine_version = 1;
    uint32_t win_w = 600;
    uint32_t win_h = 600;

    auto window = vkw::InitGLFWWindow(app_name, win_w, win_h);
    auto instance = vkw::CreateInstance(app_name, app_version, engine_name,
                                        engine_version);
    auto surface = vkw::CreateSurface(instance, window);
    auto physical_device = vkw::GetPhysicalDevices(instance).front();
    const auto surface_format = vkw::GetSurfaceFormat(physical_device, surface);

    vkw::PrintQueueFamilyProps(physical_device);

    uint32_t queue_family_idx =
            vkw::GetGraphicPresentQueueFamilyIdx(physical_device, surface);

    const uint32_t n_queues = 1;
    auto device = vkw::CreateDevice(queue_family_idx, physical_device, n_queues,
                                    true);
    auto queue = vkw::GetQueue(device, queue_family_idx);

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
    vkw::SendToDevice(device, uniform_buf_pack, &mvpc_mat[0], sizeof(mvpc_mat));

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

    vkw::AddSubpassDesc(render_pass_pack,
                        {
                                // No input attachments
                        },
                        {
                                {0, vk::ImageLayout::eColorAttachmentOptimal},
                        },
                        {1, vk::ImageLayout::eDepthStencilAttachmentOptimal});
    vkw::UpdateRenderPass(device, render_pass_pack);

    auto frame_buffer_packs = vkw::CreateFrameBuffers(device, render_pass_pack,
                                                      {nullptr, depth_img_pack},
                                                      0, swapchain_pack);

    vkw::GLSLCompiler glsl_compiler;
    auto vert_shader_module_pack = glsl_compiler.compileFromString(
            device, VERT_SOURCE, vk::ShaderStageFlagBits::eVertex);
    auto frag_shader_module_pack = glsl_compiler.compileFromString(
            device, FRAG_SOURCE, vk::ShaderStageFlagBits::eFragment);

    const size_t vertex_buf_size = CUBE_VERTICES.size() * sizeof(Vertex);
    auto vertex_buf_pack = vkw::CreateBufferPack(
            physical_device, device, vertex_buf_size,
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
    vkw::SendToDevice(device, vertex_buf_pack, CUBE_VERTICES.data(),
                      vertex_buf_size);

    vkw::PipelineInfo pipeline_info;
    pipeline_info.color_blend_infos.resize(1);
    auto pipeline_pack = vkw::CreatePipeline(
            device, {vert_shader_module_pack, frag_shader_module_pack},
            {{0, sizeof(Vertex)}},
            {{0, 0, vk::Format::eR32G32B32A32Sfloat, 0},
             {1, 0, vk::Format::eR32G32B32A32Sfloat, 16}},
            pipeline_info, {desc_set_pack}, render_pass_pack);

    const uint32_t n_cmd_bufs = 1;
    auto cmd_bufs_pack =
            vkw::CreateCommandBuffersPack(device, queue_family_idx, n_cmd_bufs);
    auto &cmd_buf = cmd_bufs_pack->cmd_bufs[0];

    // ------------------

    // Get the index of the next available swapchain image:
    auto imageAcquiredSemaphore = vkw::CreateSemaphore(device);
    const uint64_t FenceTimeout = 100000000;
    vk::ResultValue<uint32_t> currentBuffer = device->acquireNextImageKHR(
            swapchain_pack->swapchain.get(), FenceTimeout,
            imageAcquiredSemaphore->get(), nullptr);
    assert(currentBuffer.result == vk::Result::eSuccess);
    assert(currentBuffer.value < frame_buffer_packs.size());

    vkw::BeginCommand(cmd_buf);

    const std::array<float, 4> clear_color = {0.2f, 0.2f, 0.2f, 0.2f};
    vkw::AddCommandBeginRenderPass(cmd_buf, render_pass_pack,
                                   frame_buffer_packs[currentBuffer.value],
                                   {
                                           vk::ClearColorValue(clear_color),
                                           vk::ClearDepthStencilValue(1.0f, 0),
                                   });

    cmd_buf->bindPipeline(vk::PipelineBindPoint::eGraphics,
                          pipeline_pack->pipeline.get());
    cmd_buf->bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                pipeline_pack->pipeline_layout.get(), 0,
                                desc_set_pack->desc_set.get(), nullptr);

    cmd_buf->bindVertexBuffers(0, *vertex_buf_pack->buf, {0});
    cmd_buf->setViewport(
            0, vk::Viewport(0.0f, 0.0f,
                            static_cast<float>(swapchain_pack->size.width),
                            static_cast<float>(swapchain_pack->size.height),
                            0.0f, 1.0f));
    cmd_buf->setScissor(0,
                        vk::Rect2D(vk::Offset2D(0, 0), swapchain_pack->size));

    cmd_buf->draw(12 * 3, 1, 0, 0);
    // vkw::AddCommandNextSubPass(cmd_buf);
    vkw::AddCommandEndRenderPass(cmd_buf);

    vkw::EndCommand(cmd_buf);

    auto drawFence = vkw::CreateFence(device);

    vk::PipelineStageFlags waitDestinationStageMask(
            vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo(1, &imageAcquiredSemaphore->get(),
                              &waitDestinationStageMask, 1, &cmd_buf.get());

    queue.submit(submitInfo, drawFence->get());

    while (vk::Result::eTimeout ==
           device->waitForFences(drawFence->get(), VK_TRUE, FenceTimeout))
        ;

    queue.presentKHR(vk::PresentInfoKHR(0, nullptr, 1,
                                        &swapchain_pack->swapchain.get(),
                                        &currentBuffer.value));

    while (!glfwWindowShouldClose(window.get())) {
        glfwPollEvents();
    }

    std::cout << "exit" << std::endl;

    return 0;
}
