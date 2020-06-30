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

#define USE_F16_TEX

// -----------------------------------------------------------------------------
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
#ifdef USE_F16_TEX
    std::vector<uint16_t> color_tex_f16;  // 4ch
#endif

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
    std::mt19937 mt_engine(5489U);
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
#ifdef USE_F16_TEX
        ret_mesh.color_tex_f16 = vkw::CastFloat32To16(ret_mesh.color_tex);
        ret_mesh.color_tex.clear();
#endif
        // Load bump texture
        ret_mesh.bump_tex =
                LoadTextureF32(dirname + tiny_mat.bump_texname, 1,
                               &ret_mesh.bump_tex_w, &ret_mesh.bump_tex_h);
    }

    return ret_mesh;
}

// -----------------------------------------------------------------------------

class VkApp {
public:
    VkApp();
    ~VkApp();

    void initBasicComps(const vkw::WindowPtr& window);
    void initDescComps(uint32_t uniform_size, uint32_t color_tex_w,
                       uint32_t color_tex_h);
    void initAttachComps();
    void initShaderComps();
    void initVertexBuffer(const void* data, uint64_t n_bytes);
    void initPipeline();
    void initCmdBufs();
    void sendTexture(const void* tex_data, uint64_t tex_n_bytes);
    void initDrawStates(uint32_t n_vtxs,
                        const std::array<float, 4> clear_color);
    void draw(const void* uniform_data, uint64_t uniform_n_bytes);

    uint32_t getSwapchainWidth() const {
        return m_swapchain_pack->size.width;
    }
    uint32_t getSwapchainHeight() const {
        return m_swapchain_pack->size.height;
    }

private:
    const std::string APP_NAME = "VK App";
    const uint32_t APP_VERSION = 1;
    const std::string ENGINE_NAME = "VKW";
    const uint32_t ENGINE_VERSION = 1;

    const bool DEBUG_ENABLE = true;
    const bool DISPLAY_ENABLE = true;
    const uint32_t N_QUEUES = 2;

    // Basic components
    vkw::WindowPtr m_window;
    vk::UniqueInstance m_instance;
    vk::PhysicalDevice m_physical_device;
    vk::UniqueSurfaceKHR m_surface;
    vk::UniqueDevice m_device;
    vkw::SwapchainPackPtr m_swapchain_pack;
    std::vector<vk::Queue> m_queues;
    vk::Format m_surface_format;
    uint32_t m_queue_family_idx;

    // Descriptor components
    vkw::BufferPackPtr m_uniform_buf_pack;
    vkw::TexturePackPtr m_color_tex_pack;
    vkw::DescSetPackPtr m_desc_set_pack;

    // Attachment components
    const vk::Format DEPTH_FORMAT = vk::Format::eD16Unorm;
    vkw::ImagePackPtr m_depth_img_pack;
    vkw::RenderPassPackPtr m_render_pass_pack;
    std::vector<vkw::FrameBufferPackPtr> m_frame_buf_packs;

    // Shader components
    vkw::ShaderModulePackPtr m_vert_shader_pack;
    vkw::ShaderModulePackPtr m_frag_shader_pack;

    // Vertex buffers
    vkw::BufferPackPtr m_vert_buf_pack;

    // Pipelines
    vkw::PipelinePackPtr m_pipeline_pack;

    // Command buffers
    vkw::CommandBuffersPackPtr m_cmd_bufs_pack;

    // Drawing states
    std::vector<vkw::FencePtr> m_draw_fences;
    uint32_t m_prev_img_idx = uint32_t(~0);

    // Shader sources
    const std::string VERT_SOURCE = R"(
        #version 460

        layout(binding = 0) uniform UniformBuffer {
            mat4 mvp_mat;
        } uniform_buf;

        layout (location = 0) in vec3 pos;
        layout (location = 1) in lowp vec3 normal;
        layout (location = 2) in lowp vec2 uv;

        layout (location = 0) out lowp vec3 vtx_normal;
        layout (location = 1) out lowp vec2 vtx_uv;

        void main() {
            gl_Position = uniform_buf.mvp_mat * vec4(pos, 1.0);
            vtx_normal = normal;
            vtx_uv = uv;
        }
    )";
    const std::string FRAG_SOURCE = R"(
        #version 460

        layout (binding = 1) uniform sampler2D tex;

        layout (location = 0) in lowp vec3 vtx_normal;
        layout (location = 1) in lowp vec2 vtx_uv;

        layout (location = 0) out vec4 frag_color;

        void main() {
            frag_color = texture(tex, vec2(1.0 - vtx_uv.x, 1.0 - vtx_uv.y));
            // frag_color = vec4(vtx_uv, 0.0, 1.0);
            // frag_color = vec4(vtx_normal * 0.5 + 0.5, 1.0);
        }
    )";
};

VkApp::VkApp() {}

VkApp::~VkApp() {}

void VkApp::initBasicComps(const vkw::WindowPtr& window) {
    m_window = window;

    // Create instance
    m_instance =
            vkw::CreateInstance(APP_NAME, APP_VERSION, ENGINE_NAME,
                                ENGINE_VERSION, DEBUG_ENABLE, DISPLAY_ENABLE);

    // Get a physical_device
    m_physical_device = vkw::GetFirstPhysicalDevice(m_instance);

    // Create surface
    m_surface = vkw::CreateSurface(m_instance, m_window);
    m_surface_format = vkw::GetSurfaceFormat(m_physical_device, m_surface);

    // Select queue family
    m_queue_family_idx =
            vkw::GetGraphicPresentQueueFamilyIdx(m_physical_device, m_surface);
    // Create device
    m_device = vkw::CreateDevice(m_queue_family_idx, m_physical_device,
                                 N_QUEUES, DISPLAY_ENABLE);

    // Create swapchain
    m_swapchain_pack =
            vkw::CreateSwapchainPack(m_physical_device, m_device, m_surface);

    // Get queues
    m_queues.clear();
    m_queues.reserve(N_QUEUES);
    for (uint32_t i = 0; i < N_QUEUES; i++) {
        m_queues.push_back(vkw::GetQueue(m_device, m_queue_family_idx, i));
    }
}

void VkApp::initDescComps(uint32_t uniform_size, uint32_t color_tex_w,
                          uint32_t color_tex_h) {
    // Create uniform buffer
    m_uniform_buf_pack = vkw::CreateBufferPack(
            m_physical_device, m_device, uniform_size,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);

    // Create color texture
    auto color_img_pack = vkw::CreateImagePack(
#ifdef USE_F16_TEX
            m_physical_device, m_device, vk::Format::eR16G16B16A16Sfloat,
#else
            m_physical_device, m_device, vk::Format::eR32G32B32A32Sfloat,
#endif
            {color_tex_w, color_tex_h},
            vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eTransferDst,
            {}, true, vk::ImageAspectFlagBits::eColor);
    m_color_tex_pack = vkw::CreateTexturePack(color_img_pack, m_device);

    // Create descriptor set for uniform buffer and texture
    m_desc_set_pack = vkw::CreateDescriptorSetPack(
            m_device, {{vk::DescriptorType::eUniformBufferDynamic, 1,
                        vk::ShaderStageFlagBits::eVertex},
                       {vk::DescriptorType::eCombinedImageSampler, 1,
                        vk::ShaderStageFlagBits::eFragment}});

    // Bind descriptor set with actual buffer
    auto write_desc_set_pack = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack, m_desc_set_pack, 0,
                         {m_uniform_buf_pack});
    vkw::AddWriteDescSet(write_desc_set_pack, m_desc_set_pack, 1,
                         {m_color_tex_pack});
    vkw::UpdateDescriptorSets(m_device, write_desc_set_pack);
}

void VkApp::initAttachComps() {
    // Create depth buffer
    m_depth_img_pack = vkw::CreateImagePack(
            m_physical_device, m_device, DEPTH_FORMAT, m_swapchain_pack->size,
            vk::ImageUsageFlagBits::eDepthStencilAttachment, {},
            true,  // tiling
            vk::ImageAspectFlagBits::eDepth);

    // Create render pass
    m_render_pass_pack = vkw::CreateRenderPassPack();
    // Add color attachment
    vkw::AddAttachientDesc(
            m_render_pass_pack, m_surface_format, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::ePresentSrcKHR);
    // Add depth attachment
    vkw::AddAttachientDesc(m_render_pass_pack, DEPTH_FORMAT,
                           vk::AttachmentLoadOp::eClear,
                           vk::AttachmentStoreOp::eDontCare,
                           vk::ImageLayout::eDepthStencilAttachmentOptimal);
    // Add subpass
    vkw::AddSubpassDesc(m_render_pass_pack,
                        {
                                // No input attachments
                        },
                        {
                                {0, vk::ImageLayout::eColorAttachmentOptimal},
                        },
                        {1, vk::ImageLayout::eDepthStencilAttachmentOptimal});
    // Create render pass instance
    vkw::UpdateRenderPass(m_device, m_render_pass_pack);

    // Create frame buffers for swapchain images
    m_frame_buf_packs = vkw::CreateFrameBuffers(m_device, m_render_pass_pack,
                                                {nullptr, m_depth_img_pack},
                                                m_swapchain_pack);
}

void VkApp::initShaderComps() {
    // Compile shaders
    vkw::GLSLCompiler glsl_compiler;
    m_vert_shader_pack = glsl_compiler.compileFromString(
            m_device, VERT_SOURCE, vk::ShaderStageFlagBits::eVertex);
    m_frag_shader_pack = glsl_compiler.compileFromString(
            m_device, FRAG_SOURCE, vk::ShaderStageFlagBits::eFragment);
}

void VkApp::initVertexBuffer(const void* data, uint64_t n_bytes) {
    // Create vertex buffer
    m_vert_buf_pack = vkw::CreateBufferPack(
            m_physical_device, m_device, n_bytes,
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
    // Send vertices to GPU
    vkw::SendToDevice(m_device, m_vert_buf_pack, data, n_bytes);
}

void VkApp::initPipeline() {
    // Create pipeline
    vkw::PipelineInfo pipeline_info;
    pipeline_info.color_blend_infos.resize(1);
    m_pipeline_pack = vkw::CreateGraphicsPipeline(
            m_device, {m_vert_shader_pack, m_frag_shader_pack},
            {{0, sizeof(Vertex), vk::VertexInputRate::eVertex}},
            {{0, 0, vk::Format::eR32G32B32Sfloat, 0},
             {1, 0, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3},
             {2, 0, vk::Format::eR32G32Sfloat, sizeof(float) * 6}},
            pipeline_info, {m_desc_set_pack}, m_render_pass_pack);
}

void VkApp::initCmdBufs() {
    // Create command buffers
    const uint32_t n_views = static_cast<uint32_t>(m_frame_buf_packs.size());
    const bool reset_enable = false;
    m_cmd_bufs_pack = vkw::CreateCommandBuffersPack(
            m_device, m_queue_family_idx, n_views, reset_enable);
}

void VkApp::sendTexture(const void* tex_data, uint64_t tex_n_bytes) {
    // Sending command buffer for color texture
    auto send_cmd_buf_pack = vkw::CreateCommandBuffersPack(
            m_device, m_queue_family_idx, 1, false);
    auto& cmd_buf = send_cmd_buf_pack->cmd_bufs[0];

    // Create source transfer buffer
    vkw::BufferPackPtr src_trans_buf_pack = vkw::CreateBufferPack(
            m_physical_device, m_device, tex_n_bytes,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostCoherent |
                    vk::MemoryPropertyFlagBits::eHostVisible);

    // Send to buffer
    vkw::SendToDevice(m_device, src_trans_buf_pack, tex_data, tex_n_bytes);

    // Copy from buffer to image
    vkw::BeginCommand(cmd_buf);
    vkw::CopyBufferToImage(cmd_buf, src_trans_buf_pack, m_color_tex_pack);
    vkw::EndCommand(cmd_buf);

    // Send
    auto send_fence = vkw::CreateFence(m_device);
    vkw::QueueSubmit(m_queues[0], cmd_buf, send_fence, {}, {});
    vkw::WaitForFence(m_device, send_fence);
}

void VkApp::initDrawStates(uint32_t n_vtxs,
                           const std::array<float, 4> clear_color) {
    // Build commands
    auto& cmd_bufs = m_cmd_bufs_pack->cmd_bufs;
    for (size_t i = 0; i < cmd_bufs.size(); i++) {
        auto& cmd_buf = cmd_bufs[i];
        vkw::BeginCommand(cmd_buf);

        // Begin Render pass
        vkw::CmdBeginRenderPass(cmd_buf, m_render_pass_pack,
                                m_frame_buf_packs[i],
                                {
                                        vk::ClearColorValue(clear_color),
                                        vk::ClearDepthStencilValue(1.0f, 0),
                                });
        // Set pipeline
        vkw::CmdBindPipeline(cmd_buf, m_pipeline_pack);
        // Set descriptor set
        const std::vector<uint32_t> dynamic_offsets = {0};
        vkw::CmdBindDescSets(cmd_buf, m_pipeline_pack, {m_desc_set_pack},
                             dynamic_offsets);
        // Set Vertex buffer
        vkw::CmdBindVertexBuffers(cmd_buf, 0, {m_vert_buf_pack});
        // Set viewport and scissor
        vkw::CmdSetViewport(cmd_buf, m_swapchain_pack->size);
        vkw::CmdSetScissor(cmd_buf, m_swapchain_pack->size);
        // Draw
        const uint32_t n_instances = 1;
        vkw::CmdDraw(cmd_buf, n_vtxs, n_instances);
        // End Render Pass
        // vkw::CmdNextSubPass(cmd_buf);
        vkw::CmdEndRenderPass(cmd_buf);

        vkw::EndCommand(cmd_buf);
    }

    // Create asynchronous variables
    m_draw_fences.resize(cmd_bufs.size());
    for (uint32_t i = 0; i < cmd_bufs.size(); i++) {
        // Create drawing fence
        m_draw_fences[i] = vkw::CreateFence(m_device);
    }
}

void VkApp::draw(const void* uniform_data, uint64_t uniform_n_bytes) {
    // Note: For two drawing for asynchronous, create two uniform buffers or
    //       use lock properly.

    // Update uniform
    vkw::SendToDevice(m_device, m_uniform_buf_pack, uniform_data,
                      uniform_n_bytes);

    // Get next image index of swapchain
    auto img_acquired_semaphore = vkw::CreateSemaphore(m_device);
    uint32_t curr_img_idx = vkw::AcquireNextImage(
            m_device, m_swapchain_pack, img_acquired_semaphore, nullptr);

    // Emit drawing command
    auto& cmd_buf = m_cmd_bufs_pack->cmd_bufs[curr_img_idx];
    auto& draw_fence = m_draw_fences[curr_img_idx];
    vkw::QueueSubmit(m_queues[0], cmd_buf, draw_fence,
                     {{img_acquired_semaphore,
                       vk::PipelineStageFlagBits::eColorAttachmentOutput}},
                     {});

    // Present current view
    vkw::QueuePresent(m_queues[0], m_swapchain_pack, curr_img_idx, {});

    // Wait for drawing
    vkw::WaitForFence(m_device, draw_fence);
    vkw::ResetFence(m_device, draw_fence);

    // Shift image index
    m_prev_img_idx = curr_img_idx;
}

// -----------------------------------------------------------------------------

void RunExampleApp03(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    // Load mesh
#if defined(__ANDROID__)
    const std::string OBJ_FILENAME = "/sdcard/vulkanwrapper/earth/earth.obj";
#else
    const std::string OBJ_FILENAME = "../data/earth/earth.obj";
#endif
    const float OBJ_SCALE = 1.f / 100.f;
    const float SHIFT_SCALE = 30.f;
    const uint32_t N_OBJECTS = 200;
    Mesh mesh = LoadObjMany(OBJ_FILENAME, OBJ_SCALE, SHIFT_SCALE, N_OBJECTS);

    VkApp app;
    app.initBasicComps(window);
    app.initDescComps(sizeof(glm::mat4), mesh.color_tex_w, mesh.color_tex_h);
    app.initAttachComps();
    app.initShaderComps();
    app.initVertexBuffer(mesh.vertices.data(),
                         mesh.vertices.size() * sizeof(Vertex));
    app.initPipeline();
    app.initCmdBufs();
#ifdef USE_F16_TEX
    app.sendTexture(mesh.color_tex_f16.data(),
                    mesh.color_tex_f16.size() * sizeof(mesh.color_tex_f16[0]));
#else
    app.sendTexture(mesh.color_tex.data(),
                    mesh.color_tex.size() * sizeof(mesh.color_tex[0]));
#endif
    app.initDrawStates(mesh.vertices.size(), {0.2f, 0.2f, 0.2f, 1.0f});

    glm::mat4 model_mat = glm::scale(glm::vec3(1.00f));
    const glm::mat4 view_mat = glm::lookAt(glm::vec3(0.0f, 0.0f, -100.0f),
                                           glm::vec3(0.0f, 0.0f, 0.0f),
                                           glm::vec3(0.0f, 1.0f, 0.0f));
    const float aspect = static_cast<float>(app.getSwapchainWidth()) /
                         static_cast<float>(app.getSwapchainHeight());
    const glm::mat4 proj_mat =
            glm::perspective(glm::radians(90.0f), aspect, 0.1f, 1000.0f);
    // vulkan clip space has inverted y and half z !
    const glm::mat4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                                0.0f, 0.0f, 0.5f, 1.0f};

    while (true) {
        model_mat = glm::rotate(0.01f, glm::vec3(0.f, 1.f, 0.f)) * model_mat;
        const glm::mat4 mvpc_mat = clip_mat * proj_mat * view_mat * model_mat;
        app.draw(&mvpc_mat[0], sizeof(mvpc_mat));
        vkw::PrintFps();
        draw_hook();
    }
}
