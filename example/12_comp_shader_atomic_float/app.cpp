#include "app.h"

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

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
namespace {

const std::string COMP_SOURCE = R"(
#version 460
#extension GL_EXT_shader_atomic_float : require
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0, r32f) uniform readonly image2D inp_img;
layout (binding = 1, r32f) uniform image2D out_img;

void main() {
    imageAtomicAdd(out_img, ivec2(0, 0), 0.1);
}
)";

bool IsAlmostEq(const float& a, const float& b) {
    return std::abs(a - b) < 0.01;
}

}  // namespace

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void RunExampleApp12(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    (void)window;
    (void)draw_hook;

    // Initialize without display environment
    const bool display_enable = false;
    const bool debug_enable = true;
    const uint32_t n_queues = 1;

    // Create instance
    auto instance = vkw::CreateInstance("VKW Example 12", 1, "VKW", 0,
                                        debug_enable, display_enable);
    // Get a physical_device
    auto physical_device = vkw::GetFirstPhysicalDevice(instance);

    // Set features
    auto features = vkw::GetPhysicalFeatures2(physical_device);
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features;
    atomic_float_features.shaderImageFloat32AtomicAdd = true;
    features->setPNext(&atomic_float_features);

    // Select queue family
    uint32_t queue_family_idx = vkw::GetQueueFamilyIdxs(physical_device)[0];
    // Create device
    auto device = vkw::CreateDevice(queue_family_idx, physical_device, n_queues,
                                    display_enable, features,
                                    {"VK_EXT_shader_atomic_float"});

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
    const uint32_t IMG_SIZE = 4;
    std::vector<float> org_data(IMG_SIZE * IMG_SIZE);
    for (uint32_t i = 0; i < org_data.size(); i++) {
        org_data[i] = 0.0f;
    }

    // Create input image
    auto inp_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR32Sfloat,
            {IMG_SIZE, IMG_SIZE}, 1,
            vk::ImageUsageFlagBits::eStorage |
                    vk::ImageUsageFlagBits::eTransferDst,
            {}, true, vk::ImageAspectFlagBits::eColor);
    {
        // Create source buffer
        auto buf_src =
                vkw::CreateBufferPack(physical_device, device,
                                      IMG_SIZE * IMG_SIZE * sizeof(float),
                                      vk::BufferUsageFlagBits::eTransferSrc,
                                      vkw::HOST_VISIB_COHER_PROPS);
        // Send to source buffer
        vkw::SendToDevice(device, buf_src, org_data.data(),
                          org_data.size() * sizeof(float));
        // Copy from buffer to image
        vkw::BeginCommand(cmd_buf);
        vkw::CopyBufferToImage(cmd_buf, buf_src, inp_img_pack,
                               vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eGeneral);
        vkw::EndCommand(cmd_buf);
        // Execute
        auto fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, fence);
        vkw::WaitForFences(device, {fence});
    }

    // Create output image
    auto out_img_pack = vkw::CreateImagePack(
            physical_device, device, vk::Format::eR32Sfloat,
            {IMG_SIZE, IMG_SIZE}, 1,
            vk::ImageUsageFlagBits::eStorage |
                    vk::ImageUsageFlagBits::eTransferSrc,
            {}, true, vk::ImageAspectFlagBits::eColor);
    {
        // Make image layout "General"
        vkw::BeginCommand(cmd_buf);
        vkw::SetImageLayout(cmd_buf, out_img_pack, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eGeneral);
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
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 0, {inp_img_pack},
                         {vk::ImageLayout::eGeneral});
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 1, {out_img_pack},
                         {vk::ImageLayout::eGeneral});
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
        vkw::CmdBindPipeline(cmd_buf, pipeline_pack,
                             vk::PipelineBindPoint::eCompute);
        vkw::CmdBindDescSets(cmd_buf, pipeline_pack, {desc_set_pack}, {},
                             vk::PipelineBindPoint::eCompute);
        vkw::CmdDispatch(cmd_buf, IMG_SIZE, IMG_SIZE);
        vkw::EndCommand(cmd_buf);
        // Execute
        auto fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, fence);
        vkw::WaitForFences(device, {fence});
    }

    // Read result image
    std::vector<float> res_data(IMG_SIZE * IMG_SIZE);
    {
        // Create destination buffer
        auto buf_dst = vkw::CreateBufferPack(
                physical_device, device, res_data.size() * sizeof(float),
                vk::BufferUsageFlagBits::eTransferDst,
                vkw::HOST_VISIB_COHER_PROPS);
        // Copy from image to buffer
        vkw::BeginCommand(cmd_buf);
        vkw::CopyImageToBuffer(cmd_buf, out_img_pack, buf_dst,
                               vk::ImageLayout::eGeneral,
                               vkw::LAYOUT_DONT_CARE);
        vkw::EndCommand(cmd_buf);
        // Execute
        auto fence = vkw::CreateFence(device);
        vkw::QueueSubmit(queues[0], cmd_buf, fence);
        vkw::WaitForFences(device, {fence});
        // Receive from output buffer
        vkw::RecvFromDevice(device, buf_dst, res_data.data(),
                            res_data.size() * sizeof(float));
    }

    // Check answer
    std::cout << "ans: " << res_data[0] << std::endl;
    if (IsAlmostEq(res_data[0], 1.6f)) {
        std::cout << "Correct" << std::endl;
    } else {
        std::cout << "Incorrect" << std::endl;
    }
    // for (uint32_t i = 0; i < 16; i++) {
    //     std::cout << res_data[i] << std::endl;
    // }
}
