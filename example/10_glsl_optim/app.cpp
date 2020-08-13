#include "app.h"

#include <vkw/warning_suppressor.h>

BEGIN_VKW_SUPPRESS_WARNING
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
END_VKW_SUPPRESS_WARNING

#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
namespace {

const std::string SOURCE1 = R"(
#version 460

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 0) out vec3 vtx_normal;

void main() {
    gl_Position = vec4(pos, 1.0);
    vec3 tmp = normal;
    vtx_normal = tmp;
}
)";

const std::string SOURCE2 = R"(
#version 460

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 0) out vec3 vtx_normal;

void main() {
    gl_Position = vec4(pos, 1.0);
    vtx_normal = normal;
}
)";

}  // namespace
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void RunExampleApp10(const vkw::WindowPtr& window,
                     std::function<void()> draw_hook) {
    (void)window;
    (void)draw_hook;

    // Initialize without display environment
    const bool display_enable = false;
    const bool debug_enable = true;
    const uint32_t n_queues = 1;

    // Create instance
    auto instance = vkw::CreateInstance("VKW Example 10", 1, "VKW", 0,
                                        debug_enable, display_enable);
    // Get a physical_device
    auto physical_device = vkw::GetFirstPhysicalDevice(instance);

    // Select queue family
    uint32_t queue_family_idx = vkw::GetQueueFamilyIdxs(physical_device)[0];
    // Create device
    auto device = vkw::CreateDevice(queue_family_idx, physical_device, n_queues,
                                    display_enable);

    // Compile shaders (with optimization)
    vkw::GLSLCompiler glsl_compiler;
    glsl_compiler.enable_optim = true;
    glsl_compiler.enable_optim_size = true;
    auto shader_1 = glsl_compiler.compileFromString(
            device, SOURCE1, vk::ShaderStageFlagBits::eVertex);
    auto shader_2 = glsl_compiler.compileFromString(
            device, SOURCE2, vk::ShaderStageFlagBits::eVertex);

    // Compile shaders (without optimization)
    glsl_compiler.enable_optim = false;
    glsl_compiler.enable_optim_size = false;
    auto shader_1_raw = glsl_compiler.compileFromString(
            device, SOURCE1, vk::ShaderStageFlagBits::eVertex);
    auto shader_2_raw = glsl_compiler.compileFromString(
            device, SOURCE2, vk::ShaderStageFlagBits::eVertex);

    // Print SPIRV sizes
    std::stringstream ss;
    ss << "size 1: " << shader_1->spv_size << "\t";
    ss << "size 2: " << shader_2->spv_size << std::endl;
    ss << "size 1 raw: " << shader_1_raw->spv_size << "\t";
    ss << "size 2 raw: " << shader_2_raw->spv_size;
    vkw::PrintInfo(ss.str());

    if (shader_1->spv_size != shader_2->spv_size) {
        vkw::PrintErr("Compile result is something wrong (1 != 2)");
    }
    if (shader_1_raw->spv_size <= shader_1->spv_size) {
        vkw::PrintErr("Compile result is something wrong (1_raw <= 1)");
    }
    if (shader_1_raw->spv_size <= shader_2_raw->spv_size) {
        vkw::PrintErr("Compile result is something wrong (1_raw <= 2_raw)");
    }
}
