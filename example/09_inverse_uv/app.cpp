#include "app.h"

#include <vkw/warning_suppressor.h>
#include "vkw/vkw.h"
#include "vulkan/vulkan.hpp"

BEGIN_VKW_SUPPRESS_WARNING
#include <stb/stb_image.h>
#include <tinyobjloader/tiny_obj_loader.h>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
END_VKW_SUPPRESS_WARNING

#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
const std::string VERT_SOURCE1 = R"(
#version 460
layout(binding = 0) uniform UniformBuffer {
    mat4 mvp_mat;
} uniform_buf;

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 uv;

layout (location = 0) out vec2 vtx_uv;

void main() {
    gl_Position = uniform_buf.mvp_mat * vec4(pos, 1.0);
    vtx_uv = uv;
}
)";
const std::string FRAG_SOURCE1 = R"(
#version 460
layout (location = 0) in vec2 vtx_uv;
layout (location = 0) out vec4 frag_color;

void main() {
    frag_color = vec4(vtx_uv, 0.0, 1.0);
}
)";

const uint32_t COMP_LOCAL_SIZE = 4;  // TODO
const std::string COMP_SOURCE = R"(
#version 460
layout (local_size_x = 4, local_size_y = 4) in;
layout (binding = 0, rg16f) uniform readonly image2D inp_img;
layout (binding = 1, rg16f) uniform writeonly image2D out_img;

void main() {
    uvec2 screen_size = gl_WorkGroupSize.xy * gl_NumWorkGroups.xy;
    ivec2 ss_pos = ivec2(gl_GlobalInvocationID.xy);
    vec2 ss_pos_f = vec2(ss_pos) / (screen_size * 0.5);
    vec2 uv_f = imageLoad(inp_img, ss_pos).xy;
    ivec2 uv = ivec2(uv_f * screen_size);
    imageStore(out_img, uv, vec4(ss_pos_f, 1.0, 1.0));
}
)";

const std::string VERT_SOURCE2 = R"(
#version 460

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 screen_pos = uv * 2.0f - 1.0f;
    gl_Position = vec4(screen_pos, 0.0f, 1.0f);
}
)";
const std::string FRAG_SOURCE2 = R"(
#version 460
layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inp_img;
layout (location = 0) out vec4 frag_color;

void main() {
    vec4 col = subpassLoad(inp_img);
    frag_color = col;
}
)";

struct Vertex {
    float x, y, z;     // Position
    float u, v;        // Texcoord
};

struct Mesh {
    std::vector<Vertex> vertices;
};

static Mesh LoadObj(const std::string& filename, const float scale) {
    // Load with tiny obj
    tinyobj::ObjReader obj_reader;
    const bool ret = obj_reader.ParseFromFile(filename);
    if (!ret) {
        std::stringstream ss;
        ss << "Error:" << obj_reader.Error() << std::endl;
        ss << "Warning:" << obj_reader.Warning() << std::endl;
        throw std::runtime_error(ss.str());
    }
    const std::vector<tinyobj::shape_t>& tiny_shapes = obj_reader.GetShapes();
    const tinyobj::attrib_t& tiny_attrib = obj_reader.GetAttrib();
    const std::vector<tinyobj::real_t>& tiny_vertices = tiny_attrib.vertices;
    const std::vector<tinyobj::real_t>& tiny_texcoords = tiny_attrib.texcoords;

    // Parse to mesh structure
    Mesh ret_mesh;
    for (const tinyobj::shape_t& tiny_shape : tiny_shapes) {
        const tinyobj::mesh_t& tiny_mesh = tiny_shape.mesh;
        for (const tinyobj::index_t& tiny_idx : tiny_mesh.indices) {
            // Parse one vertex
            Vertex ret_vtx = {};
            if (0 <= tiny_idx.vertex_index) {
                auto idx0 = static_cast<uint32_t>(tiny_idx.vertex_index * 3);
                ret_vtx.x = tiny_vertices[idx0 + 0] * scale;
                ret_vtx.y = tiny_vertices[idx0 + 1] * scale;
                ret_vtx.z = tiny_vertices[idx0 + 2] * scale;
            }
            if (0 <= tiny_idx.texcoord_index) {
                auto idx0 = static_cast<uint32_t>(tiny_idx.texcoord_index * 2);
                ret_vtx.u = tiny_texcoords[idx0 + 0];
                ret_vtx.v = tiny_texcoords[idx0 + 1];
            }
            // Register
            ret_mesh.vertices.push_back(ret_vtx);
        }
    }

    return ret_mesh;
}

// -----------------------------------------------------------------------------

void RunExampleApp09(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    // Load mesh
#if defined(__ANDROID__)
    const std::string OBJ_FILENAME = "/sdcard/vulkanwrapper/earth/earth.obj";
#else
    const std::string OBJ_FILENAME = "../data/earth/earth.obj";
#endif
    const float OBJ_SCALE = 1.f / 100.f;
    Mesh mesh = LoadObj(OBJ_FILENAME, OBJ_SCALE);

    // Initialize with display environment
    const bool display_enable = true;
    const bool debug_enable = true;
    const uint32_t n_queues = 1;

    // Create instance
    auto instance = vkw::CreateInstance("VKW Example 09", 1, "VKW", 0,
                                        debug_enable, display_enable);

    // Get a physical_device
    auto physical_device = vkw::GetFirstPhysicalDevice(instance);

    // Create surface
    auto surface = vkw::CreateSurface(instance, window);
    auto surface_format = vkw::GetSurfaceFormat(physical_device, surface);

    // Select queue family
    uint32_t queue_family_idx =
            vkw::GetGraphicPresentQueueFamilyIdx(physical_device, surface);
    // Create device
    auto device = vkw::CreateDevice(queue_family_idx, physical_device, n_queues,
                                    display_enable);

    // Create swapchain
    auto swapchain_pack =
            vkw::CreateSwapchainPack(physical_device, device, surface);

    // Get queues
    std::vector<vk::Queue> queues;
    queues.reserve(n_queues);
    for (uint32_t i = 0; i < n_queues; i++) {
        queues.push_back(vkw::GetQueue(device, queue_family_idx, i));
    }

    // Create depth buffer
    const auto depth_format = vk::Format::eD16Unorm;
    auto depth_img_pack = vkw::CreateImagePack(
            physical_device, device, depth_format, swapchain_pack->size,
            vk::ImageUsageFlagBits::eDepthStencilAttachment, {},
            true,  // tiling
            vk::ImageAspectFlagBits::eDepth);

    // UV image (mid)
    auto uv_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR16G16Sfloat,
            swapchain_pack->size,
            vk::ImageUsageFlagBits::eStorage |
                    vk::ImageUsageFlagBits::eColorAttachment,
            {}, true, vk::ImageAspectFlagBits::eColor);

    // Result image
    auto result_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR16G16Sfloat,
            swapchain_pack->size,
            vk::ImageUsageFlagBits::eStorage |
                    vk::ImageUsageFlagBits::eTransferDst |
                    vk::ImageUsageFlagBits::eInputAttachment,
            {}, true, vk::ImageAspectFlagBits::eColor);

    // Create uniform buffer
    auto uniform_buf_pack = vkw::CreateBufferPack(
            physical_device, device, sizeof(glm::mat4),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);

    // Create render pass (1st)
    auto render_pass_pack1 = vkw::CreateRenderPassPack();
    // Add color attachment
    vkw::AddAttachientDesc(
            render_pass_pack1, uv_img_pack->format, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::eGeneral);
    // Add depth attachment
    vkw::AddAttachientDesc(render_pass_pack1, depth_format,
                           vk::AttachmentLoadOp::eClear,
                           vk::AttachmentStoreOp::eDontCare,
                           vk::ImageLayout::eDepthStencilAttachmentOptimal);
    // Add subpass
    vkw::AddSubpassDesc(render_pass_pack1,
                        {
                                // No input attachments
                        },
                        {
                                {0, vk::ImageLayout::eGeneral},
                        },
                        {1, vk::ImageLayout::eDepthStencilAttachmentOptimal});
    // Create render pass instance
    vkw::UpdateRenderPass(device, render_pass_pack1);
    // Create frame buffers for swapchain images
    auto frame_buffer_pack1 =
            vkw::CreateFrameBuffer(device, render_pass_pack1,
                                    {uv_img_pack, depth_img_pack});

    // Create descriptor set for uniform buffer and texture
    auto desc_set_pack1 = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eUniformBufferDynamic, 1,
                      vk::ShaderStageFlagBits::eVertex}});
    // Bind descriptor set with actual buffer
    auto write_desc_set_pack1 = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack1, desc_set_pack1, 0,
                         {uniform_buf_pack});
    vkw::UpdateDescriptorSets(device, write_desc_set_pack1);

    // Create render pass (2nd)
    auto render_pass_pack2 = vkw::CreateRenderPassPack();
    // Add input attachment
    vkw::AddAttachientDesc(
            render_pass_pack2, result_img_pack->format,
            vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore,
            vk::ImageLayout::eGeneral);
    // Add color attachment
    vkw::AddAttachientDesc(
            render_pass_pack2, surface_format, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::ePresentSrcKHR);
    // Add subpass
    vkw::AddSubpassDesc(render_pass_pack2,
                        {
                                {0, vk::ImageLayout::eGeneral},
                        },
                        {
                                {1, vk::ImageLayout::eColorAttachmentOptimal},
                        });  // no depth
    // Create render pass instance
    vkw::UpdateRenderPass(device, render_pass_pack2);
    // Create frame buffers for swapchain images
    auto frame_buffer_packs2 =
            vkw::CreateFrameBuffers(device, render_pass_pack2,
                                    {result_img_pack, nullptr},
                                    swapchain_pack);

    // Create descriptor set for uniform buffer and texture
    auto desc_set_pack2 = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eInputAttachment, 1,
                      vk::ShaderStageFlagBits::eFragment}});
    // Bind descriptor set with actual buffer
    auto write_desc_set_pack2 = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack2, desc_set_pack2, 0,
                         {result_img_pack}, {vk::ImageLayout::eGeneral});
    vkw::UpdateDescriptorSets(device, write_desc_set_pack2);

    // Create descriptor set for compute shader
    auto desc_set_pack_comp = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eStorageImage, 1,
                      vk::ShaderStageFlagBits::eCompute},  // Input image
                     {vk::DescriptorType::eStorageImage, 1,
                      vk::ShaderStageFlagBits::eCompute}});  // Output image
    auto write_desc_set_pack_comp = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack_comp, desc_set_pack_comp, 0,
                         {uv_img_pack}, {vk::ImageLayout::eGeneral});
    vkw::AddWriteDescSet(write_desc_set_pack_comp, desc_set_pack_comp, 1,
                         {result_img_pack}, {vk::ImageLayout::eGeneral});
    vkw::UpdateDescriptorSets(device, write_desc_set_pack_comp);

    // Compile shaders
    vkw::GLSLCompiler glsl_compiler;
    auto vert_shader_module_pack1 = glsl_compiler.compileFromString(
            device, VERT_SOURCE1, vk::ShaderStageFlagBits::eVertex);
    auto frag_shader_module_pack1 = glsl_compiler.compileFromString(
            device, FRAG_SOURCE1, vk::ShaderStageFlagBits::eFragment);
    auto vert_shader_module_pack2 = glsl_compiler.compileFromString(
            device, VERT_SOURCE2, vk::ShaderStageFlagBits::eVertex);
    auto frag_shader_module_pack2 = glsl_compiler.compileFromString(
            device, FRAG_SOURCE2, vk::ShaderStageFlagBits::eFragment);
    auto comp_shader_module_pack = glsl_compiler.compileFromString(
            device, COMP_SOURCE, vk::ShaderStageFlagBits::eCompute);

    // Create vertex buffer
    const size_t vertex_buf_size = mesh.vertices.size() * sizeof(Vertex);
    auto vertex_buf_pack = vkw::CreateBufferPack(
            physical_device, device, vertex_buf_size,
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
    // Send vertices to GPU
    vkw::SendToDevice(device, vertex_buf_pack, mesh.vertices.data(),
                      vertex_buf_size);

    // Create pipeline (1st)
    vkw::PipelineInfo pipeline_info1;
    pipeline_info1.color_blend_infos.resize(1);
    auto pipeline_pack1 = vkw::CreateGraphicsPipeline(
            device, {vert_shader_module_pack1, frag_shader_module_pack1},
            {{0, sizeof(Vertex), vk::VertexInputRate::eVertex}},
            {{0, 0, vk::Format::eR32G32B32Sfloat, 0},
             {1, 0, vk::Format::eR32G32Sfloat, sizeof(float) * 3}},
            pipeline_info1, {desc_set_pack1}, render_pass_pack1);

    // Create pipeline (2nd)
    vkw::PipelineInfo pipeline_info2;
    pipeline_info2.color_blend_infos.resize(1);
    auto pipeline_pack2 = vkw::CreateGraphicsPipeline(
            device, {vert_shader_module_pack2, frag_shader_module_pack2},
            {}, {}, pipeline_info2, {desc_set_pack2}, render_pass_pack2);

    // Create pipeline (compute shader)
    vkw::PipelineInfo pipeline_info;
    auto pipeline_pack_comp = vkw::CreateComputePipeline(
            device, comp_shader_module_pack, {desc_set_pack_comp});

    const uint32_t n_cmd_bufs = 1;
    auto cmd_bufs_pack =
            vkw::CreateCommandBuffersPack(device, queue_family_idx, n_cmd_bufs);
    auto& cmd_buf = cmd_bufs_pack->cmd_bufs[0];

    // ------------------
    glm::mat4 model_mat = glm::scale(glm::vec3(1.00f));
    const glm::mat4 view_mat = glm::lookAt(glm::vec3(0.0f, 0.0f, -10.0f),
                                           glm::vec3(0.0f, 0.0f, 0.0f),
                                           glm::vec3(0.0f, 1.0f, 0.0f));
    const float aspect = static_cast<float>(swapchain_pack->size.width) /
                         static_cast<float>(swapchain_pack->size.height);
    const glm::mat4 proj_mat =
            glm::perspective(glm::radians(45.0f), aspect, 0.1f, 1000.0f);
    // vulkan clip space has inverted y and half z !
    const glm::mat4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                                0.0f, 0.0f, 0.5f, 1.0f};

    while (true) {
        model_mat = glm::rotate(0.01f, glm::vec3(0.f, 1.f, 0.f)) * model_mat;
        glm::mat4 mvpc_mat = clip_mat * proj_mat * view_mat * model_mat;
        vkw::SendToDevice(device, uniform_buf_pack, &mvpc_mat[0],
                          sizeof(mvpc_mat));

        const std::array<float, 4> clear_color = {0.0f, 0.0f, 0.0f, 1.0f};

        auto img_acquired_semaphore = vkw::CreateSemaphore(device);
        uint32_t curr_img_idx = vkw::AcquireNextImage(
                device, swapchain_pack, img_acquired_semaphore, nullptr);

        vkw::ResetCommand(cmd_buf);

        vkw::BeginCommand(cmd_buf);
        // Layout initialization
        vkw::SetImageLayout(cmd_buf, uv_img_pack, vk::ImageLayout::eGeneral);
        vkw::SetImageLayout(cmd_buf, result_img_pack, vk::ImageLayout::eGeneral);
        vkw::ClearColorImage(cmd_buf, result_img_pack,
                             vk::ClearColorValue(clear_color));

        // 1st pass
        vkw::CmdBeginRenderPass(cmd_buf, render_pass_pack1,
                                frame_buffer_pack1,
                                {
                                        vk::ClearColorValue(clear_color),
                                        vk::ClearDepthStencilValue(1.0f, 0),
                                });
        vkw::CmdBindPipeline(cmd_buf, pipeline_pack1);
        const std::vector<uint32_t> dynamic_offsets = {0};
        vkw::CmdBindDescSets(cmd_buf, pipeline_pack1, {desc_set_pack1},
                             dynamic_offsets);
        vkw::CmdBindVertexBuffers(cmd_buf, 0, {vertex_buf_pack});
        vkw::CmdSetViewport(cmd_buf, swapchain_pack->size);
        vkw::CmdSetScissor(cmd_buf, swapchain_pack->size);
        vkw::CmdDraw(cmd_buf, mesh.vertices.size());
        vkw::CmdEndRenderPass(cmd_buf);

        // Compute pass
        vkw::CmdBindPipeline(cmd_buf, pipeline_pack_comp,
                             vk::PipelineBindPoint::eCompute);
        vkw::CmdBindDescSets(cmd_buf, pipeline_pack_comp, {desc_set_pack_comp},
                             {}, vk::PipelineBindPoint::eCompute);
        vkw::CmdDispatch(cmd_buf, swapchain_pack->size.width / COMP_LOCAL_SIZE,
                         swapchain_pack->size.height / COMP_LOCAL_SIZE);

        // 2nd pass
        vkw::CmdBeginRenderPass(cmd_buf, render_pass_pack2,
                                frame_buffer_packs2[curr_img_idx],
                                {
                                        vk::ClearColorValue(clear_color),
                                        vk::ClearDepthStencilValue(1.0f, 0),
                                });
        vkw::CmdBindPipeline(cmd_buf, pipeline_pack2);
        vkw::CmdBindDescSets(cmd_buf, pipeline_pack2, {desc_set_pack2});
        vkw::CmdSetViewport(cmd_buf, swapchain_pack->size);
        vkw::CmdSetScissor(cmd_buf, swapchain_pack->size);
        vkw::CmdDraw(cmd_buf, 3);
        vkw::CmdEndRenderPass(cmd_buf);

        vkw::EndCommand(cmd_buf);
        vkw::FencePtr draw_fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, {draw_fence},
                         {{img_acquired_semaphore,
                           vk::PipelineStageFlagBits::eColorAttachmentOutput}},
                         {});
        vkw::QueuePresent(queues[0], swapchain_pack, curr_img_idx);
        vkw::WaitForFence(device, draw_fence);

        // end
        vkw::PrintFps();
        draw_hook();
    }
}
