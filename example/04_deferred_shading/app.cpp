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
const std::string VERT_SOURCE = R"(
#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (std140, binding = 0) uniform buffer {
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

const std::string FRAG_SOURCE = R"(
#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 1) uniform sampler2D tex;

layout (location = 0) in vec3 vtx_normal;
layout (location = 1) in vec2 vtx_uv;

layout (location = 0) out vec4 frag_color;
layout (location = 1) out vec4 frag_normal;

void main() {
    frag_color = texture(tex, vec2(1.0 - vtx_uv.x, 1.0 - vtx_uv.y));
//     frag_color = vec4(vtx_uv, 0.0, 1.0);
    frag_normal = vec4(vtx_normal * 0.5 + 0.5, 1.0);
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

// -----------------------------------------------------------------------------

void RunExampleApp04(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    // Load mesh
#if defined(__ANDROID__)
    const std::string& OBJ_FILENAME = "/sdcard/vulkanwrapper/earth/earth.obj";
#else
    const std::string& OBJ_FILENAME = "../data/earth/earth.obj";
#endif
    const float OBJ_SCALE = 1.f / 100.f;
    const float SHIFT_SCALE = 100.f;
    const uint32_t N_OBJECTS = 400;
    Mesh mesh = LoadObjMany(OBJ_FILENAME, OBJ_SCALE, SHIFT_SCALE, N_OBJECTS);

    // Initialize with display environment
    const bool display_enable = true;
    const bool debug_enable = false;
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

    // Create depth buffer
    const auto depth_format = vk::Format::eD16Unorm;
    auto depth_img_pack = vkw::CreateImagePack(
            physical_device, device, depth_format, swapchain_pack->size,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::ImageAspectFlagBits::eDepth, true, false);

    // Create extra buffer
//     const auto extra_format = vk::Format::eA8B8G8R8UintPack32;
    const auto extra_format = surface_format;
    auto extra_img_pack = vkw::CreateImagePack(
            physical_device, device, extra_format, swapchain_pack->size,
            vk::ImageUsageFlagBits::eColorAttachment,
            vk::ImageAspectFlagBits::eColor, true, false);

    // Create uniform buffer
    auto uniform_buf_pack = vkw::CreateBufferPack(
            physical_device, device, sizeof(glm::mat4),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);

    // Create color texture
    auto color_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR32G32B32A32Sfloat,
            {mesh.color_tex_w, mesh.color_tex_h},
            vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor,
            true, false);
    auto color_tex_pack = vkw::CreateTexturePack(color_img_pack, device);

    // Create bump texture
    auto bump_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR32Sfloat,
            {mesh.color_tex_w, mesh.color_tex_h},
            vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor,
            true, false);
    auto bump_tex_pack = vkw::CreateTexturePack(bump_img_pack, device);

    // Create descriptor set for uniform buffer and texture
    auto desc_set_pack = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eUniformBufferDynamic, 1,
                      vk::ShaderStageFlagBits::eVertex},
                     {vk::DescriptorType::eCombinedImageSampler, 2,
                      vk::ShaderStageFlagBits::eFragment}});

    // Bind descriptor set with actual buffer
    auto write_desc_set_pack = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 0,
                         {uniform_buf_pack});
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 1,
                         {color_tex_pack, bump_tex_pack});
    vkw::UpdateDescriptorSets(device, write_desc_set_pack);

    // Create render pass
    auto render_pass_pack = vkw::CreateRenderPassPack();
    // Add color attachment for surface
    vkw::AddAttachientDesc(
            render_pass_pack, surface_format, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::ePresentSrcKHR);
    // Add color extra attachment
    vkw::AddAttachientDesc(
            render_pass_pack, extra_format, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::ePresentSrcKHR);
//             vk::AttachmentStoreOp::eStore, vk::ImageLayout::eColorAttachmentOptimal);
    // Add depth attachment
    vkw::AddAttachientDesc(render_pass_pack, depth_format,
                           vk::AttachmentLoadOp::eClear,
                           vk::AttachmentStoreOp::eDontCare,
                           vk::ImageLayout::eDepthStencilAttachmentOptimal);
    // Add subpass
    vkw::AddSubpassDesc(render_pass_pack,
                        {
                                // No input attachments
                        },
                        {
                                {0, vk::ImageLayout::eColorAttachmentOptimal},
                                {1, vk::ImageLayout::eColorAttachmentOptimal},
                        },
                        {2, vk::ImageLayout::eDepthStencilAttachmentOptimal});
    // Create render pass instance
    vkw::UpdateRenderPass(device, render_pass_pack);

    // Create frame buffers for swapchain images
    auto frame_buffer_packs = vkw::CreateFrameBuffers(device, render_pass_pack,
//                                                       {extra_img_pack, nullptr, depth_img_pack},
//                                                       swapchain_pack);
                                                      {nullptr, extra_img_pack, depth_img_pack},
                                                      swapchain_pack);

    // Compile shaders
    vkw::GLSLCompiler glsl_compiler;
    auto vert_shader_module_pack = glsl_compiler.compileFromString(
            device, VERT_SOURCE, vk::ShaderStageFlagBits::eVertex);
    auto frag_shader_module_pack = glsl_compiler.compileFromString(
            device, FRAG_SOURCE, vk::ShaderStageFlagBits::eFragment);

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

    // Create pipeline
    vkw::PipelineInfo pipeline_info;
    pipeline_info.color_blend_infos.resize(2);
    auto pipeline_pack = vkw::CreatePipeline(
            device, {vert_shader_module_pack, frag_shader_module_pack},
            {{0, sizeof(Vertex), vk::VertexInputRate::eVertex}},
            {{0, 0, vk::Format::eR32G32B32Sfloat, 0},
             {1, 0, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3},
             {2, 0, vk::Format::eR32G32Sfloat, sizeof(float) * 6}},
            pipeline_info, {desc_set_pack}, render_pass_pack);

    uint32_t n_cmd_bufs = static_cast<uint32_t>(frame_buffer_packs.size());
    auto cmd_bufs_pack =
            vkw::CreateCommandBuffersPack(device, queue_family_idx, n_cmd_bufs);
    auto& cmd_bufs = cmd_bufs_pack->cmd_bufs;

    // ------------------
    {
        // Send color texture to GPU
        auto& cmd_buf = cmd_bufs[0];
        vkw::BeginCommand(cmd_buf);
        vkw::SendToDevice(
                device, color_tex_pack, mesh.color_tex.data(),
                mesh.color_tex.size() * sizeof(mesh.color_tex[0]), cmd_buf);
        vkw::EndCommand(cmd_buf);
        auto send_fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, send_fence, {}, {});
        vkw::WaitForFence(device, send_fence);

        // Sending buffer will not be used anymore.
        color_tex_pack->img_pack->trans_buf_pack.reset();
    }
    {
        // Send color texture to GPU
        auto& cmd_buf = cmd_bufs[0];
        vkw::BeginCommand(cmd_buf);
        vkw::SendToDevice(
                device, bump_tex_pack, mesh.bump_tex.data(),
                mesh.bump_tex.size() * sizeof(mesh.bump_tex[0]), cmd_buf);
        vkw::EndCommand(cmd_buf);
        auto send_fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, send_fence, {}, {});
        vkw::WaitForFence(device, send_fence);

        // Sending buffer will not be used anymore.
        bump_tex_pack->img_pack->trans_buf_pack.reset();
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

    for (int render_idx = 0; render_idx < 60 * 10; render_idx++) {
        model_mat = glm::rotate(0.01f, glm::vec3(0.f, 1.f, 0.f)) * model_mat;
        glm::mat4 mvpc_mat = clip_mat * proj_mat * view_mat * model_mat;
        vkw::SendToDevice(device, uniform_buf_pack, &mvpc_mat[0],
                          sizeof(mvpc_mat));

        auto img_acquired_semaphore = vkw::CreateSemaphore(device);
        uint32_t curr_img_idx = 0;
        vkw::AcquireNextImage(&curr_img_idx, device, swapchain_pack,
                              img_acquired_semaphore, nullptr);

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
            vkw::CmdBeginRenderPass(cmd_buf, render_pass_pack,
                                    frame_buffer_packs[curr_img_idx],
                                    {
                                            vk::ClearColorValue(clear_color),
                                            vk::ClearColorValue(clear_color),
                                            vk::ClearDepthStencilValue(1.0f, 0),
                                    });

            vkw::CmdBindPipeline(cmd_buf, pipeline_pack);

            const std::vector<uint32_t> dynamic_offsets = {0};
            vkw::CmdBindDescSets(cmd_buf, pipeline_pack, {desc_set_pack},
                                 dynamic_offsets);

            vkw::CmdBindVertexBuffers(cmd_buf, {vertex_buf_pack});

            vkw::CmdSetViewport(cmd_buf, swapchain_pack->size);
            vkw::CmdSetScissor(cmd_buf, swapchain_pack->size);

            const uint32_t n_instances = 1;
            vkw::CmdDraw(cmd_buf, mesh.vertices.size(), n_instances);

            // vkw::CmdNextSubPass(cmd_buf);
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
