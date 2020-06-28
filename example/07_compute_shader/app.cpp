#include "app.h"

#include <vkw/warning_suppressor.h>

BEGIN_VKW_SUPPRESS_WARNING
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
END_VKW_SUPPRESS_WARNING

// -----------------------------------------------------------------------------
const std::string VERT_SOURCE = R"(
#version 460

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 color;

layout (location = 0) out vec4 vtx_color;
layout (location = 1) out float vtx_id_f;

layout(binding = 0) uniform UniformBuffer {
    mat4 mvp;
} uniformBuffer;

void main() {
    gl_Position = uniformBuffer.mvp * pos;
    vtx_color = color;
    vtx_id_f = float(gl_VertexIndex) + 0.5;
}
)";


// -----------------------------------------------------------------------------

void RunExampleApp07(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    // Initialize with display environment
    const bool display_enable = true;
    const bool debug_enable = true;
    const uint32_t n_queues = 2;

//     // Create instance
//     auto instance = vkw::CreateInstance("VKW Example 07", 1, "VKW", 0,
//                                         debug_enable, display_enable);
//     // Get a physical_device
//     auto physical_device = vkw::GetFirstPhysicalDevice(instance);
// 
//     // Set features
//     auto features = vkw::GetPhysicalFeatures(physical_device);
//     features->geometryShader = true;
// 
//     // Create surface
//     auto surface = vkw::CreateSurface(instance, window);
//     auto surface_format = vkw::GetSurfaceFormat(physical_device, surface);
// 
//     // Select queue family
//     uint32_t queue_family_idx =
//             vkw::GetGraphicPresentQueueFamilyIdx(physical_device, surface);
//     // Create device
//     auto device = vkw::CreateDevice(queue_family_idx, physical_device, n_queues,
//                                     display_enable, features);
// 
//     // Create swapchain
//     auto swapchain_pack =
//             vkw::CreateSwapchainPack(physical_device, device, surface);
// 
//     // Get queues
//     std::vector<vk::Queue> queues;
//     queues.reserve(n_queues);
//     for (uint32_t i = 0; i < n_queues; i++) {
//         queues.push_back(vkw::GetQueue(device, queue_family_idx, i));
//     }
// 
//     // Create depth buffer
//     const auto depth_format = vk::Format::eD16Unorm;
//     auto depth_img_pack = vkw::CreateImagePack(
//             physical_device, device, depth_format, swapchain_pack->size,
//             vk::ImageUsageFlagBits::eDepthStencilAttachment,
//             vk::ImageAspectFlagBits::eDepth, true, false);
// 
//     // Create uniform buffer
//     auto uniform_buf_pack = vkw::CreateBufferPack(
//             physical_device, device, sizeof(glm::mat4),
//             vk::BufferUsageFlagBits::eUniformBuffer,
//             vk::MemoryPropertyFlagBits::eHostVisible |
//                     vk::MemoryPropertyFlagBits::eHostCoherent);
// 
//     // Create description set
//     auto desc_set_pack = vkw::CreateDescriptorSetPack(
//             device, {{vk::DescriptorType::eUniformBufferDynamic, 1,
//                       vk::ShaderStageFlagBits::eVertex}});
//     // Set actual buffer to descriptor set
//     auto write_desc_set_pack = vkw::CreateWriteDescSetPack();
//     vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 0,
//                          {uniform_buf_pack});
//     vkw::UpdateDescriptorSets(device, write_desc_set_pack);
// 
//     // Create render pass
//     auto render_pass_pack = vkw::CreateRenderPassPack();
//     vkw::AddAttachientDesc(
//             render_pass_pack, surface_format, vk::AttachmentLoadOp::eClear,
//             vk::AttachmentStoreOp::eStore, vk::ImageLayout::ePresentSrcKHR);
//     vkw::AddAttachientDesc(render_pass_pack, depth_format,
//                            vk::AttachmentLoadOp::eClear,
//                            vk::AttachmentStoreOp::eDontCare,
//                            vk::ImageLayout::eDepthStencilAttachmentOptimal);
// 
//     // Create subpass
//     vkw::AddSubpassDesc(render_pass_pack,
//                         {
//                                 // No input attachments
//                         },
//                         {
//                                 {0, vk::ImageLayout::eColorAttachmentOptimal},
//                         },
//                         {1, vk::ImageLayout::eDepthStencilAttachmentOptimal});
//     vkw::UpdateRenderPass(device, render_pass_pack);
// 
//     // Create frame buffer
//     auto frame_buffer_packs = vkw::CreateFrameBuffers(device, render_pass_pack,
//                                                       {nullptr, depth_img_pack},
//                                                       swapchain_pack);
// 
//     // Compile shader
//     vkw::GLSLCompiler glsl_compiler;
//     auto vert_shader_module_pack = glsl_compiler.compileFromString(
//             device, VERT_SOURCE, vk::ShaderStageFlagBits::eVertex);
//     auto geom_shader_module_pack = glsl_compiler.compileFromString(
//             device, GEOM_SOURCE, vk::ShaderStageFlagBits::eGeometry);
//     auto frag_shader_module_pack = glsl_compiler.compileFromString(
//             device, FRAG_SOURCE, vk::ShaderStageFlagBits::eFragment);
// 
//     // Create vertex buffer
//     const size_t vertex_buf_size = CUBE_VERTICES.size() * sizeof(Vertex);
//     auto vertex_buf_pack = vkw::CreateBufferPack(
//             physical_device, device, vertex_buf_size,
//             vk::BufferUsageFlagBits::eVertexBuffer,
//             vk::MemoryPropertyFlagBits::eHostVisible |
//                     vk::MemoryPropertyFlagBits::eHostCoherent);
//     vkw::SendToDevice(device, vertex_buf_pack, CUBE_VERTICES.data(),
//                       vertex_buf_size);
// 
//     // Create index buffer
//     const size_t index_buf_size = CUBE_INDICES.size() * sizeof(uint32_t);
//     auto index_buf_pack = vkw::CreateBufferPack(
//             physical_device, device, index_buf_size,
//             vk::BufferUsageFlagBits::eIndexBuffer,
//             vk::MemoryPropertyFlagBits::eHostVisible |
//                     vk::MemoryPropertyFlagBits::eHostCoherent);
//     vkw::SendToDevice(device, index_buf_pack, CUBE_INDICES.data(),
//                       index_buf_size);
// 
//     // Create pipeline
//     vkw::PipelineInfo pipeline_info;
//     pipeline_info.color_blend_infos.resize(1);
//     auto pipeline_pack = vkw::CreatePipeline(
//             device, {vert_shader_module_pack, geom_shader_module_pack,
//                      frag_shader_module_pack},
//             {{0, sizeof(Vertex), vk::VertexInputRate::eVertex}},
//             {{0, 0, vk::Format::eR32G32B32A32Sfloat, 0},
//              {1, 0, vk::Format::eR32G32B32A32Sfloat, 16}},
//             pipeline_info, {desc_set_pack}, render_pass_pack);
// 
//     const uint32_t n_cmd_bufs = 1;
//     auto cmd_bufs_pack =
//             vkw::CreateCommandBuffersPack(device, queue_family_idx, n_cmd_bufs);
//     auto& cmd_buf = cmd_bufs_pack->cmd_bufs[0];
// 
//     // ------------------
//     const glm::mat4 model_mat = glm::mat4(1.0f);
//     const glm::mat4 view_mat = glm::lookAt(glm::vec3(-5.0f, 3.0f, -10.0f),
//                                            glm::vec3(0.0f, 0.0f, 0.0f),
//                                            glm::vec3(0.0f, -1.0f, 0.0f));
//     const float aspect = static_cast<float>(swapchain_pack->size.width) /
//                          static_cast<float>(swapchain_pack->size.height);
//     const glm::mat4 proj_mat =
//             glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
//     // vulkan clip space has inverted y and half z !
//     const glm::mat4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
//                                 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
//                                 0.0f, 0.0f, 0.5f, 1.0f};
//     glm::mat4 rot_mat(1.f);
// 
//     while (true) {
//         rot_mat = glm::rotate(0.01f, glm::vec3(1.f, 0.f, 0.f)) * rot_mat;
//         glm::mat4 mvpc_mat =
//                 clip_mat * proj_mat * view_mat * rot_mat * model_mat;
//         vkw::SendToDevice(device, uniform_buf_pack, &mvpc_mat[0],
//                           sizeof(mvpc_mat));
// 
//         vkw::ResetCommand(cmd_buf);
// 
//         auto img_acquired_semaphore = vkw::CreateSemaphore(device);
//         uint32_t curr_img_idx = vkw::AcquireNextImage(
//                 device, swapchain_pack, img_acquired_semaphore, nullptr);
// 
//         vkw::BeginCommand(cmd_buf);
// 
//         const std::array<float, 4> clear_color = {0.2f, 0.2f, 0.2f, 0.2f};
//         vkw::CmdBeginRenderPass(cmd_buf, render_pass_pack,
//                                 frame_buffer_packs[curr_img_idx],
//                                 {
//                                         vk::ClearColorValue(clear_color),
//                                         vk::ClearDepthStencilValue(1.0f, 0),
//                                 });
// 
//         vkw::CmdBindPipeline(cmd_buf, pipeline_pack);
// 
//         const std::vector<uint32_t> dynamic_offsets = {0};
//         vkw::CmdBindDescSets(cmd_buf, pipeline_pack, {desc_set_pack},
//                              dynamic_offsets);
// 
//         vkw::CmdBindVertexBuffers(cmd_buf, 0, {vertex_buf_pack});
//         vkw::CmdBindIndexBuffer(cmd_buf, index_buf_pack, 0,
//                                 vk::IndexType::eUint32);
// 
//         vkw::CmdSetViewport(cmd_buf, swapchain_pack->size);
//         vkw::CmdSetScissor(cmd_buf, swapchain_pack->size);
// 
//         const uint32_t n_instances = 1;
//         vkw::CmdDrawIndexed(cmd_buf, static_cast<uint32_t>(CUBE_INDICES.size()),
//                             n_instances);
// 
//         // vkw::CmdNextSubPass(cmd_buf);
//         vkw::CmdEndRenderPass(cmd_buf);
// 
//         vkw::EndCommand(cmd_buf);
// 
//         auto draw_fence = vkw::CreateFence(device);
// 
//         vkw::QueueSubmit(queues[0], cmd_buf, draw_fence,
//                          {{img_acquired_semaphore,
//                            vk::PipelineStageFlagBits::eColorAttachmentOutput}},
//                          {});
// 
//         vkw::QueuePresent(queues[1], swapchain_pack, curr_img_idx);
// 
//         vkw::WaitForFences(device, {draw_fence});
// 
//         vkw::PrintFps();
// 
//         draw_hook();
//     }
}