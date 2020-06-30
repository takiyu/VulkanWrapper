#include "app.h"

#include <bits/stdint-uintn.h>
#include <vkw/warning_suppressor.h>

#include "vkw/vkw.h"
#include "vulkan/vulkan.hpp"

BEGIN_VKW_SUPPRESS_WARNING
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
END_VKW_SUPPRESS_WARNING

#include <iostream>

const std::string COMP_SOURCE = R"(
#version 460
layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, rgba32f) uniform readonly image2D inp_img;
layout (binding = 1, rgba32f) uniform writeonly image2D out_img;

vec4 invCol(in vec4 c) {
    return c.bgra;
}

void main() {
    vec4 col = imageLoad(inp_img, ivec2(gl_GlobalInvocationID.xy));
    vec4 result = invCol(col);
    imageStore(out_img, ivec2(gl_GlobalInvocationID.xy), result);
}
)";

bool IsAlmostEq(const float &a, const float &b) {
    return std::abs(a - b) < 0.01;
}

// -----------------------------------------------------------------------------
void RunExampleApp08(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    (void)window;
    (void)draw_hook;

    // Initialize without display environment
    const bool display_enable = false;
    const bool debug_enable = true;
    const uint32_t n_queues = 1;

    // Create instance
    auto instance = vkw::CreateInstance("VKW Example 08", 1, "VKW", 0,
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

    // -------------------------------------------------------------------------
    // Create original CPU data
    const uint32_t IMG_SIZE = 256;
    std::vector<float> org_data(IMG_SIZE * IMG_SIZE * 4);
    for (uint32_t i = 0; i < org_data.size() / 4; i++) {
        org_data[i * 4 + 0] = 0.0f;
        org_data[i * 4 + 1] = 0.3f;
        org_data[i * 4 + 2] = 0.7f;
        org_data[i * 4 + 3] = 1.0f;
    }

    // Create input image
    auto inp_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR32G32B32A32Sfloat,
            {IMG_SIZE, IMG_SIZE},
            vk::ImageUsageFlagBits::eStorage |
                    vk::ImageUsageFlagBits::eTransferDst,
            {}, true, vk::ImageAspectFlagBits::eColor);
    {
        // Create source buffer
        auto buf_src =
                vkw::CreateBufferPack(physical_device, device,
                                      IMG_SIZE * IMG_SIZE * 4 * sizeof(float),
                                      vk::BufferUsageFlagBits::eTransferSrc,
                                      vkw::HOST_VISIB_COHER_PROPS);
        // Send to source buffer
        vkw::SendToDevice(device, buf_src, org_data.data(),
                          org_data.size() * sizeof(float));
        // Copy from buffer to image
        vkw::BeginCommand(cmd_buf);
        vkw::CopyBufferToImage(cmd_buf, buf_src, inp_img_pack,
                               vk::ImageLayout::eGeneral);
        vkw::EndCommand(cmd_buf);
        // Execute
        auto fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, fence);
        vkw::WaitForFences(device, {fence});
    }

    // Create output image
    auto out_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR32G32B32A32Sfloat,
            {IMG_SIZE, IMG_SIZE},
            vk::ImageUsageFlagBits::eStorage |
                    vk::ImageUsageFlagBits::eTransferSrc,
            {}, true, vk::ImageAspectFlagBits::eColor);
    {
        // Make image layout "General"
        vkw::BeginCommand(cmd_buf);
        vkw::SetImageLayout(cmd_buf, out_img_pack, vk::ImageLayout::eGeneral);
        vkw::EndCommand(cmd_buf);
        // Execute
        auto fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, fence);
        vkw::WaitForFences(device, {fence});
    }

    // Create descriptor set
    auto desc_set_pack = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eStorageImage, 1,
                      vk::ShaderStageFlagBits::eCompute},  // Input image
                     {vk::DescriptorType::eStorageImage, 1,
                      vk::ShaderStageFlagBits::eCompute}});  // Output image
    auto write_desc_set_pack = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 0,
                         {inp_img_pack});
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 1,
                         {out_img_pack});
    vkw::UpdateDescriptorSets(device, write_desc_set_pack);

    // Compile shader
    vkw::GLSLCompiler glsl_compiler;
    auto comp_shader_module_pack = glsl_compiler.compileFromString(
            device, COMP_SOURCE, vk::ShaderStageFlagBits::eCompute);

    // Create pipeline
    vkw::PipelineInfo pipeline_info;
    auto pipeline_pack = vkw::CreateComputePipeline(
            device, comp_shader_module_pack, {desc_set_pack});

    // Dispatch
    {
        // Make image layout "General"
        vkw::BeginCommand(cmd_buf);
        vkw::CmdBindPipeline(cmd_buf, pipeline_pack, vk::PipelineBindPoint::eCompute);
        vkw::CmdBindDescSets(cmd_buf, pipeline_pack, {desc_set_pack}, {},
                             vk::PipelineBindPoint::eCompute);
        vkw::CmdDispatch(cmd_buf, IMG_SIZE / 16, IMG_SIZE / 16);
        vkw::EndCommand(cmd_buf);
        // Execute
        auto fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, fence);
        vkw::WaitForFences(device, {fence});
    }

    // Read result image
    std::vector<float> res_data(IMG_SIZE * IMG_SIZE * 4);
    {
        // Create destination buffer
        auto buf_dst = vkw::CreateBufferPack(physical_device, device, res_data.size() * sizeof(float),
                    vk::BufferUsageFlagBits::eTransferDst, vkw::HOST_VISIB_COHER_PROPS);
        // Copy from image to buffer
        vkw::BeginCommand(cmd_buf);
        vkw::CopyImageToBuffer(cmd_buf, out_img_pack, buf_dst, vk::ImageLayout::eGeneral);
        vkw::EndCommand(cmd_buf);
        // Execute
        auto fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, fence);
        vkw::WaitForFences(device, {fence});
        // Receive from output buffer
        vkw::RecvFromDevice(device, buf_dst, res_data.data(), res_data.size() * sizeof(float));
    }

    // Check answer
    bool is_all_correct = true;
    for (uint32_t i = 0; i < res_data.size() / 4; i++) {
        if (IsAlmostEq(res_data[i * 4 + 0], org_data[i * 4 + 2]) &&
            IsAlmostEq(res_data[i * 4 + 1], org_data[i * 4 + 1]) &&
            IsAlmostEq(res_data[i * 4 + 2], org_data[i * 4 + 0]) &&
            IsAlmostEq(res_data[i * 4 + 3], org_data[i * 4 + 3])) {
            continue;
        } else {
            is_all_correct = false;
            vkw::PrintErr("Computation result is wrong.");
        }
    }
    if (is_all_correct) {
        vkw::PrintInfo("Compute shader is success.");
    }
}
