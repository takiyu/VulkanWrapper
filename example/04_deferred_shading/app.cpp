#include "app.h"

#include <vkw/warning_suppressor.h>

BEGIN_VKW_SUPPRESS_WARNING
#include <stb/stb_image.h>
#include <tinyobjloader/tiny_obj_loader.h>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
END_VKW_SUPPRESS_WARNING

#include <iostream>
#include <random>
#include <sstream>

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
namespace {

const std::string VERT_SOURCE1 = R"(
#version 460

layout (binding = 0) uniform UniformBuffer {
    mat4 mvp_mat;
} uniform_buf;

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec3 vtx_normal;
layout (location = 1) out vec2 vtx_uv;

void main() {
    gl_Position = uniform_buf.mvp_mat * vec4(pos, 1.0);
    vtx_normal = normal;
    vtx_uv = uv;
}
)";
const std::string FRAG_SOURCE1 = R"(
#version 460

layout (set=0, binding = 1) uniform sampler2D texs[2];

layout (location = 0) in vec3 vtx_normal;
layout (location = 1) in vec2 vtx_uv;

layout (location = 0) out vec4 frag_colors[2];

void main() {
    // Texture fetching
    vec2 uv = vec2(1.0 - vtx_uv.x, 1.0 - vtx_uv.y);
    vec4 color = texture(texs[0], uv);
    float bump = texture(texs[1], uv).r;

    frag_colors[0] = color;
    frag_colors[1] = vec4(vtx_normal, 1.0);
}
)";

const std::string VERT_SOURCE2 = R"(
#version 460

layout (location = 0) out vec2 vtx_uv;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 screen_pos = uv * 2.0f - 1.0f;
    gl_Position = vec4(screen_pos, 0.0f, 1.0f);
    vtx_uv = uv;
}
)";
const std::string FRAG_SOURCE2 = R"(
#version 460

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput input_imgs[2];

layout (location = 0) in vec2 vtx_uv;

layout (location = 0) out vec4 frag_color;

void main() {
    vec4 v0 = subpassLoad(input_imgs[0]).rgba;
    vec4 v1 = subpassLoad(input_imgs[1]).rgba;
    frag_color = mix(v0, v1, step(vtx_uv.x, 0.5));
}
)";

struct Vertex {
    float x, y, z;     // Position
    float nx, ny, nz;  // Normal
    float u, v;        // Texcoord
};

struct Mesh {
    std::vector<Vertex> vertices;

    uint32_t color_tex_w = 0;
    uint32_t color_tex_h = 0;
    std::vector<float> color_tex;  // 4ch

    uint32_t bump_tex_w = 0;
    uint32_t bump_tex_h = 0;
    std::vector<float> bump_tex;  // 1ch
};

static std::string ExtractDirname(const std::string& path) {
    return path.substr(0, path.find_last_of('/') + 1);
}

static std::vector<float> LoadTextureF32(const std::string& filename,
                                         const uint32_t n_ch, uint32_t* w,
                                         uint32_t* h) {
    // Load image using STB
    int tmp_w, tmp_h, dummy_c;
    uint8_t* data = stbi_load(filename.c_str(), &tmp_w, &tmp_h, &dummy_c,
                              static_cast<int>(n_ch));
    (*w) = static_cast<uint32_t>(tmp_w);
    (*h) = static_cast<uint32_t>(tmp_h);

    // Cast to float
    std::vector<float> ret_tex_f32;
    const size_t n_pix = (*w) * (*h) * n_ch;
    ret_tex_f32.reserve(n_pix);
    for (size_t idx = 0; idx < n_pix; idx++) {
        const uint8_t& v = *(data + idx);
        ret_tex_f32.push_back(v / 255.f);
    }

    // Release STB memory
    stbi_image_free(data);

    return ret_tex_f32;
}

static Mesh LoadObjMany(const std::string& filename, const float obj_scale,
                        const float shift_scale, const uint32_t n_objects) {
    const std::string& dirname = ExtractDirname(filename);

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
    const std::vector<tinyobj::real_t>& tiny_normals = tiny_attrib.normals;
    const std::vector<tinyobj::real_t>& tiny_texcoords = tiny_attrib.texcoords;

    // Deterministic random number generator
    std::random_device rnd;
    std::mt19937 mt_engine(0);
    std::uniform_int_distribution<> distrib(-shift_scale, shift_scale);

    // Parse to mesh structure
    Mesh ret_mesh;
    for (uint32_t obj_idx = 0; obj_idx < n_objects; obj_idx++) {
        const float sx = distrib(mt_engine);
        const float sy = distrib(mt_engine);
        const float sz = distrib(mt_engine);

        for (const tinyobj::shape_t& tiny_shape : tiny_shapes) {
            const tinyobj::mesh_t& tiny_mesh = tiny_shape.mesh;
            for (const tinyobj::index_t& tiny_idx : tiny_mesh.indices) {
                // Parse one vertex
                Vertex ret_vtx = {};
                if (0 <= tiny_idx.vertex_index) {
                    auto i = static_cast<uint32_t>(tiny_idx.vertex_index * 3);
                    ret_vtx.x = tiny_vertices[i + 0] * obj_scale + sx;
                    ret_vtx.y = tiny_vertices[i + 1] * obj_scale + sy;
                    ret_vtx.z = tiny_vertices[i + 2] * obj_scale + sz;
                }
                if (0 <= tiny_idx.normal_index) {
                    auto i = static_cast<uint32_t>(tiny_idx.normal_index * 3);
                    ret_vtx.nx = tiny_normals[i + 0];
                    ret_vtx.ny = tiny_normals[i + 1];
                    ret_vtx.nz = tiny_normals[i + 2];
                }
                if (0 <= tiny_idx.texcoord_index) {
                    auto i = static_cast<uint32_t>(tiny_idx.texcoord_index * 2);
                    ret_vtx.u = tiny_texcoords[i + 0];
                    ret_vtx.v = tiny_texcoords[i + 1];
                }
                // Register
                ret_mesh.vertices.push_back(ret_vtx);
            }
        }
    }

    // Load textures
    const auto& tiny_mats = obj_reader.GetMaterials();
    if (!tiny_mats.empty()) {
        // Supports only 1 materials
        const tinyobj::material_t tiny_mat = tiny_mats[0];
        // Load color texture
        ret_mesh.color_tex =
                LoadTextureF32(dirname + tiny_mat.diffuse_texname, 4,
                               &ret_mesh.color_tex_w, &ret_mesh.color_tex_h);
        // Load bump texture
        ret_mesh.bump_tex =
                LoadTextureF32(dirname + tiny_mat.bump_texname, 1,
                               &ret_mesh.bump_tex_w, &ret_mesh.bump_tex_h);
    }

    return ret_mesh;
}

}  // namespace
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void RunExampleApp04(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    // Load mesh
#if defined(__ANDROID__)
    const std::string OBJ_FILENAME = "/sdcard/vulkanwrapper/earth/earth.obj";
#else
    const std::string OBJ_FILENAME = "../data/earth/earth.obj";
#endif
    const float OBJ_SCALE = 1.f / 100.f;
    const float SHIFT_SCALE = 100.f;
    const uint32_t N_OBJECTS = 400;
    Mesh mesh = LoadObjMany(OBJ_FILENAME, OBJ_SCALE, SHIFT_SCALE, N_OBJECTS);

    // Initialize with display environment
    const bool display_enable = true;
    const bool debug_enable = true;
    const uint32_t n_queues = 2;

    // Create instance
    auto instance = vkw::CreateInstance("VKW Example 04", 1, "VKW", 0,
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

    // Create G buffer 0 (color)
    const auto gbuf_col_format = vk::Format::eA8B8G8R8SnormPack32;
    auto gbuf_col_img_pack = vkw::CreateImagePack(
            physical_device, device, gbuf_col_format, swapchain_pack->size, 1,
            vk::ImageUsageFlagBits::eColorAttachment |
                    vk::ImageUsageFlagBits::eInputAttachment,
            {}, true, vk::ImageAspectFlagBits::eColor);

    // Create G buffer 1 (normal)
    const auto gbuf_nor_format = vk::Format::eA8B8G8R8SnormPack32;
    auto gbuf_nor_img_pack = vkw::CreateImagePack(
            physical_device, device, gbuf_nor_format, swapchain_pack->size, 1,
            vk::ImageUsageFlagBits::eColorAttachment |
                    vk::ImageUsageFlagBits::eInputAttachment,
            {}, true, vk::ImageAspectFlagBits::eColor);

    // Create depth buffer
    const auto depth_format = vk::Format::eD16Unorm;
    auto depth_img_pack = vkw::CreateImagePack(
            physical_device, device, depth_format, swapchain_pack->size, 1,
            vk::ImageUsageFlagBits::eDepthStencilAttachment, {}, true,
            vk::ImageAspectFlagBits::eDepth);

    // Create uniform buffer
    auto uniform_buf_pack = vkw::CreateBufferPack(
            physical_device, device, sizeof(glm::mat4),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);

    // Create color texture
    auto color_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR32G32B32A32Sfloat,
            {mesh.color_tex_w, mesh.color_tex_h}, 1,
            vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eTransferDst,
            {}, true, vk::ImageAspectFlagBits::eColor);
    auto color_tex_pack = vkw::CreateTexturePack(color_img_pack, device);

    // Create bump texture
    auto bump_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR32Sfloat,
            {mesh.color_tex_w, mesh.color_tex_h}, 1,
            vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eTransferDst,
            {}, true, vk::ImageAspectFlagBits::eColor);
    auto bump_tex_pack = vkw::CreateTexturePack(bump_img_pack, device);

    // Create descriptor set 0 for uniform buffer and texture
    auto desc_set_pack0 = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eUniformBufferDynamic, 1,
                      vk::ShaderStageFlagBits::eVertex},
                     {vk::DescriptorType::eCombinedImageSampler, 2,
                      vk::ShaderStageFlagBits::eFragment}});
    // Bind descriptor set with actual buffer
    auto write_desc_set_pack0 = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack0, desc_set_pack0, 0,
                         {uniform_buf_pack});
    vkw::AddWriteDescSet(write_desc_set_pack0, desc_set_pack0, 1,
                         {color_tex_pack, bump_tex_pack},  // layout is undef.
                         {vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal});
    vkw::UpdateDescriptorSets(device, write_desc_set_pack0);

    // Create descriptor set 1 for uniform buffer and texture
    auto desc_set_pack1 = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eInputAttachment, 2,
                      vk::ShaderStageFlagBits::eFragment}});
    // Bind descriptor set with actual buffer
    auto write_desc_set_pack1 = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack1, desc_set_pack1, 0,
                         {gbuf_col_img_pack, gbuf_nor_img_pack},  // no layouts
                         {vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal});
    vkw::UpdateDescriptorSets(device, write_desc_set_pack1);

    // Create render pass
    auto render_pass_pack = vkw::CreateRenderPassPack();
    // 0) Add color attachment for surface
    vkw::AddAttachientDesc(
            render_pass_pack, surface_format, vk::ImageLayout::eUndefined,
            vk::ImageLayout::ePresentSrcKHR, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore);
    // 1) Add gbuffer 0 (color) attachment
    vkw::AddAttachientDesc(
            render_pass_pack, gbuf_col_format, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore);
    // 2) Add gbuffer 1 (normal) attachment
    vkw::AddAttachientDesc(
            render_pass_pack, gbuf_nor_format, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore);
    // 3) Add depth attachment
    vkw::AddAttachientDesc(
            render_pass_pack, depth_format, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare);
    // Add subpass 1
    vkw::AddSubpassDesc(render_pass_pack,
                        {
                                // No input attachments
                        },
                        {
                                {1, vk::ImageLayout::eColorAttachmentOptimal},
                                {2, vk::ImageLayout::eColorAttachmentOptimal},
                        },
                        {3, vk::ImageLayout::eDepthStencilAttachmentOptimal});
    // Add subpass 2
    vkw::AddSubpassDesc(render_pass_pack,
                        {
                                {1, vk::ImageLayout::eShaderReadOnlyOptimal},
                                {2, vk::ImageLayout::eShaderReadOnlyOptimal},
                        },
                        {
                                {0, vk::ImageLayout::eColorAttachmentOptimal},
                        });  // No depth
    // Add dependency
    vkw::AddSubpassDepend(render_pass_pack,
                          {0, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                           vk::AccessFlagBits::eColorAttachmentWrite},
                          {1, vk::PipelineStageFlagBits::eFragmentShader,
                           vk::AccessFlagBits::eInputAttachmentRead},
                          vk::DependencyFlagBits::eByRegion);
    // Create render pass instance
    vkw::UpdateRenderPass(device, render_pass_pack);

    // Create frame buffers for swapchain images
    auto frame_buffer_packs = vkw::CreateFrameBuffers(
            device, render_pass_pack,
            {nullptr, gbuf_col_img_pack, gbuf_nor_img_pack, depth_img_pack},
            swapchain_pack);

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

    // Create pipeline 0
    vkw::PipelineInfo pipeline_info0;
    pipeline_info0.color_blend_infos.resize(2);
    auto pipeline_pack0 = vkw::CreateGraphicsPipeline(
            device, {vert_shader_module_pack1, frag_shader_module_pack1},
            {{0, sizeof(Vertex), vk::VertexInputRate::eVertex}},
            {{0, 0, vk::Format::eR32G32B32Sfloat, 0},
             {1, 0, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3},
             {2, 0, vk::Format::eR32G32Sfloat, sizeof(float) * 6}},
            pipeline_info0, {desc_set_pack0}, render_pass_pack, 0);

    // Create pipeline 1
    vkw::PipelineInfo pipeline_info1;
    pipeline_info1.color_blend_infos.resize(1);
    pipeline_info1.depth_test_enable = false;
    auto pipeline_pack1 = vkw::CreateGraphicsPipeline(
            device, {vert_shader_module_pack2, frag_shader_module_pack2}, {},
            {}, pipeline_info1, {desc_set_pack1}, render_pass_pack, 1);

    uint32_t n_cmd_bufs = static_cast<uint32_t>(frame_buffer_packs.size());
    auto cmd_bufs_pack =
            vkw::CreateCommandBuffersPack(device, queue_family_idx, n_cmd_bufs);
    auto& cmd_bufs = cmd_bufs_pack->cmd_bufs;

    // ------------------
    {
        // Send color texture to GPU
        auto& cmd_buf = cmd_bufs[0];
        const uint64_t tex_n_bytes =
                mesh.color_tex.size() * sizeof(mesh.color_tex[0]);
        vkw::BufferPackPtr src_trans_buf_pack = vkw::CreateBufferPack(
                physical_device, device, tex_n_bytes,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostCoherent |
                        vk::MemoryPropertyFlagBits::eHostVisible);
        vkw::SendToDevice(device, src_trans_buf_pack, mesh.color_tex.data(),
                          tex_n_bytes);
        vkw::BeginCommand(cmd_buf);
        vkw::CopyBufferToImage(cmd_buf, src_trans_buf_pack,
                               color_tex_pack->img_pack,
                               vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
        vkw::EndCommand(cmd_buf);
        auto send_fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, send_fence, {}, {});
        vkw::WaitForFence(device, send_fence);
    }
    {
        // Send color texture to GPU
        auto& cmd_buf = cmd_bufs[0];
        const uint64_t tex_n_bytes =
                mesh.bump_tex.size() * sizeof(mesh.bump_tex[0]);
        vkw::BufferPackPtr src_trans_buf_pack = vkw::CreateBufferPack(
                physical_device, device, tex_n_bytes,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostCoherent |
                        vk::MemoryPropertyFlagBits::eHostVisible);
        vkw::SendToDevice(device, src_trans_buf_pack, mesh.bump_tex.data(),
                          tex_n_bytes);
        vkw::BeginCommand(cmd_buf);
        vkw::CopyBufferToImage(cmd_buf, src_trans_buf_pack,
                               bump_tex_pack->img_pack,
                               vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
        vkw::EndCommand(cmd_buf);
        auto send_fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, send_fence, {}, {});
        vkw::WaitForFence(device, send_fence);
    }

    // ------------------
    glm::mat4 model_mat = glm::scale(glm::vec3(1.00f));
    const glm::mat4 view_mat = glm::lookAt(glm::vec3(0.0f, 0.0f, -100.0f),
                                           glm::vec3(0.0f, 0.0f, 0.0f),
                                           glm::vec3(0.0f, 1.0f, 0.0f));
    const float aspect = static_cast<float>(swapchain_pack->size.width) /
                         static_cast<float>(swapchain_pack->size.height);
    const glm::mat4 proj_mat =
            glm::perspective(glm::radians(90.0f), aspect, 0.1f, 1000.0f);
    // vulkan clip space has inverted y and half z !
    const glm::mat4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                                0.0f, 0.0f, 0.5f, 1.0f};

    vkw::FencePtr draw_fence = vkw::CreateFence(device);

    for (uint32_t render_idx = 0; render_idx < n_cmd_bufs * 2; render_idx++) {
        render_idx = std::min(render_idx, n_cmd_bufs);

        model_mat = glm::rotate(0.01f, glm::vec3(0.f, 1.f, 0.f)) * model_mat;
        glm::mat4 mvpc_mat = clip_mat * proj_mat * view_mat * model_mat;
        vkw::SendToDevice(device, uniform_buf_pack, &mvpc_mat[0],
                          sizeof(mvpc_mat));

        auto img_acquired_semaphore = vkw::CreateSemaphore(device);
        uint32_t curr_img_idx = vkw::AcquireNextImage(
                device, swapchain_pack, img_acquired_semaphore, nullptr);

#if 1
        auto& cmd_buf = cmd_bufs[curr_img_idx];
        if (render_idx < n_cmd_bufs) {
#elif 0
        auto& cmd_buf = cmd_bufs[curr_img_idx];
        {
#elif 0
        auto& cmd_buf = cmd_bufs[0];
        if (0 < render_idx) {
            vkw::WaitForFence(device, draw_fence);
            vkw::ResetFence(device, draw_fence);
        }
        {
#endif
            // Create command buffer
            vkw::ResetCommand(cmd_buf);
            vkw::BeginCommand(cmd_buf);

            const std::array<float, 4> clear_color = {0.2f, 0.2f, 0.2f, 1.0f};
            const std::array<float, 4> clear_normal = {0.0f, 0.0f, 0.0f, 0.0f};
            vkw::CmdBeginRenderPass(cmd_buf, render_pass_pack,
                                    frame_buffer_packs[curr_img_idx],
                                    {
                                            vk::ClearColorValue(clear_color),
                                            vk::ClearColorValue(clear_color),
                                            vk::ClearColorValue(clear_normal),
                                            vk::ClearDepthStencilValue(1.0f, 0),
                                    });

            // 1st subpass
            vkw::CmdBindPipeline(cmd_buf, pipeline_pack0);

            const std::vector<uint32_t> dynamic_offsets0 = {0};
            vkw::CmdBindDescSets(cmd_buf, pipeline_pack0, {desc_set_pack0},
                                 dynamic_offsets0);

            vkw::CmdBindVertexBuffers(cmd_buf, 0, {vertex_buf_pack});

            vkw::CmdSetViewport(cmd_buf, swapchain_pack->size);
            vkw::CmdSetScissor(cmd_buf, swapchain_pack->size);

            const uint32_t n_instances = 1;
            vkw::CmdDraw(cmd_buf, mesh.vertices.size(), n_instances);

            // 2nd subpass
            vkw::CmdNextSubPass(cmd_buf);

            vkw::CmdBindPipeline(cmd_buf, pipeline_pack1);
            vkw::CmdBindDescSets(cmd_buf, pipeline_pack1, {desc_set_pack1});
            vkw::CmdDraw(cmd_buf, 3, 1);

            vkw::CmdEndRenderPass(cmd_buf);
            vkw::EndCommand(cmd_buf);
        }

        vkw::QueueSubmit(queues[0], cmd_buf, draw_fence,
                         {{img_acquired_semaphore,
                           vk::PipelineStageFlagBits::eColorAttachmentOutput}},
                         {});

        vkw::QueuePresent(queues[0], swapchain_pack, curr_img_idx);

#if 0
        vkw::WaitForFence(device, draw_fence);
#elif 1
        vkw::WaitForFence(device, draw_fence);
        vkw::ResetFence(device, draw_fence);
#endif

        vkw::PrintFps();

        draw_hook();
    }

    draw_fence.reset();
}
