#include <vkw/vkw.h>

// -----------------------------------------------------------------------------
// ------------------------- Begin third party include -------------------------
// -----------------------------------------------------------------------------
BEGIN_VKW_SUPPRESS_WARNING
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>
#if defined(__ANDROID__)
#include <android/log.h>
#endif
END_VKW_SUPPRESS_WARNING
// -----------------------------------------------------------------------------
// -------------------------- End third party include --------------------------
// -----------------------------------------------------------------------------

#include <chrono>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

// Storage for dispatcher
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace vkw {

namespace {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
template <typename T>
inline T Clamp(const T &x, const T &min_v, const T &max_v) {
    return std::min(std::max(x, min_v), max_v);
}

template <typename BitType>
bool IsSufficient(const vk::Flags<BitType> &actual_flags,
                  const vk::Flags<BitType> &require_flags) {
    return (actual_flags & require_flags) == require_flags;
}

template <typename T>
T &EmplaceBackEmpty(std::vector<T> &vec) {
    vec.emplace_back();
    return vec.back();
}

template <typename T>
const T *DataSafety(const std::vector<T> &vec) {
    if (vec.empty()) {
        return nullptr;
    } else {
        return vec.data();
    }
}

template <typename T>
T *DataSafety(std::vector<T> &vec) {
    if (vec.empty()) {
        return nullptr;
    } else {
        return vec.data();
    }
}

static std::vector<std::string> Split(const std::string &str, char del = '\n') {
    std::vector<std::string> result;
    std::string::size_type first_pos = 0, last_pos = 0;
    while (first_pos < str.size()) {
        // Find next splitter position
        last_pos = str.find_first_of(del, first_pos);
        if (last_pos == std::string::npos) {
            // Add last item
            std::string sub_str(str, first_pos);
            result.push_back(sub_str);
            break;
        }
        // Extract sub string
        std::string sub_str(str, first_pos, last_pos - first_pos);
        result.push_back(sub_str);
        // Go to next position
        first_pos = last_pos + 1;
    }
    return result;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static bool IsVkDebugUtilsAvailable() {
    const std::string DEBUG_UTIL_NAME = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    for (auto &&prop : vk::enumerateInstanceExtensionProperties()) {
        if (std::string(prop.extensionName) == DEBUG_UTIL_NAME) {
            return true;
        }
    }
    return false;
}

static std::vector<char const *> GetEnabledLayers(bool debug_enable) {
    std::vector<char const *> names;

    if (debug_enable) {
#if defined(__ANDROID__)
        names.push_back("VK_LAYER_LUNARG_parameter_validation");
        names.push_back("VK_LAYER_GOOGLE_unique_objects");
        names.push_back("VK_LAYER_GOOGLE_threading");
        names.push_back("VK_LAYER_LUNARG_object_tracker");
        names.push_back("VK_LAYER_LUNARG_core_validation");
#else
        names.push_back("VK_LAYER_KHRONOS_validation");
#endif
    }

    // Check layer name validities
    std::set<std::string> valid_names;
    for (auto &&prop : vk::enumerateInstanceLayerProperties()) {
        if (prop.layerName) {
            valid_names.insert(std::string(prop.layerName));
        }
    }
    std::vector<char const *> ret_names;
    for (auto &&name : names) {
        if (valid_names.count(name)) {
            ret_names.push_back(name);
        } else {
            PrintErr("[Error] Layer '" + std::string(name) + "' is invalid");
        }
    }

    return ret_names;
}

static std::vector<char const *> GetEnabledExts(bool debug_enable,
                                                bool surface_enable) {
    std::vector<char const *> enabled_exts;

    if (surface_enable) {
#if defined(__ANDROID__)
        // Add android surface extensions
        enabled_exts.push_back("VK_KHR_surface");
        enabled_exts.push_back("VK_KHR_android_surface");
#else
        // Add extension names required by GLFW
        uint32_t n_glfw_ext = 0;
        const char **glfw_exts = glfwGetRequiredInstanceExtensions(&n_glfw_ext);
        for (uint32_t i = 0; i < n_glfw_ext; i++) {
            enabled_exts.push_back(glfw_exts[i]);
        }
#endif
    }

    if (debug_enable) {
        // If available, use `debug utils`
        if (IsVkDebugUtilsAvailable()) {
            enabled_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        } else {
            enabled_exts.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }
    }
    return enabled_exts;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 DebugMessengerCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
        VkDebugUtilsMessageTypeFlagsEXT msg_types,
        VkDebugUtilsMessengerCallbackDataEXT const *callback, void *) {
    // Create corresponding strings
    const std::string &severity_str =
            vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(
                    msg_severity));
    const std::string &type_str = vk::to_string(
            static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(msg_types));

    // Create message string
    std::stringstream ss;
    ss << "-----------------------------------------------" << std::endl;
    ss << severity_str << ": " << type_str << ":" << std::endl;
    ss << "  Message ID Name (number) = <" << callback->pMessageIdName << "> ("
       << callback->messageIdNumber << ")" << std::endl;
    ss << "  Message = \"" << callback->pMessage << "\"" << std::endl;
    if (0 < callback->queueLabelCount) {
        ss << "  Queue Labels:" << std::endl;
        for (uint8_t i = 0; i < callback->queueLabelCount; i++) {
            const auto &name = callback->pQueueLabels[i].pLabelName;
            ss << "    " << i << ": " << name << std::endl;
        }
    }
    if (0 < callback->cmdBufLabelCount) {
        ss << "  CommandBuffer Labels:" << std::endl;
        for (uint8_t i = 0; i < callback->cmdBufLabelCount; i++) {
            const auto &name = callback->pCmdBufLabels[i].pLabelName;
            ss << "    " << i << ": " << name << std::endl;
        }
    }
    if (0 < callback->objectCount) {
        ss << "  Objects:" << std::endl;
        for (uint8_t i = 0; i < callback->objectCount; i++) {
            const auto &type = vk::to_string(static_cast<vk::ObjectType>(
                    callback->pObjects[i].objectType));
            const auto &handle = callback->pObjects[i].objectHandle;
            ss << "    " << static_cast<int>(i) << ":" << std::endl;
            ss << "      objectType   = " << type << std::endl;
            ss << "      objectHandle = " << handle << std::endl;
            if (callback->pObjects[i].pObjectName) {
                const auto &on = callback->pObjects[i].pObjectName;
                ss << "      objectName   = <" << on << ">" << std::endl;
            }
        }
    }
    ss << "-----------------------------------------------";

    // Print error
    PrintErr(ss.str());

    return VK_TRUE;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugReportCallback(
        VkDebugReportFlagsEXT msg_flags, VkDebugReportObjectTypeEXT obj_type,
        uint64_t src_object, size_t location, int32_t msg_code,
        const char *layer_prefix, const char *message, void *) {
    (void)src_object;
    (void)location;

    // Create message string
    std::stringstream ss;
    ss << vk::to_string(vk::DebugReportFlagBitsEXT(msg_flags));
    ss << std::endl;
    ss << "  Layer: " << layer_prefix << "]";
    ss << ", Code: " << msg_code;
    ss << ", Object: " << vk::to_string(vk::DebugReportObjectTypeEXT(obj_type));
    ss << std::endl;
    ss << "  Message: " << message;

    // Print error
    PrintErr(ss.str());

    return VK_TRUE;
}

static void RegisterDebugCalback(const vk::UniqueInstance &instance) {
    // If available, use `debug utils`
    if (IsVkDebugUtilsAvailable()) {
        // Create debug messenger (only warning and error)
        vk::UniqueDebugUtilsMessengerEXT debug_messenger =
                instance->createDebugUtilsMessengerEXTUnique(
                        {{},
                         {vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                          vk::DebugUtilsMessageSeverityFlagBitsEXT::eError},
                         {vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation},
                         &DebugMessengerCallback});
    } else {
        // Create debug report (only warning and error)
        vk::UniqueDebugReportCallbackEXT debug_report =
                instance->createDebugReportCallbackEXTUnique(
                        {{vk::DebugReportFlagBitsEXT::eWarning |
                          vk::DebugReportFlagBitsEXT::eError |
                          vk::DebugReportFlagBitsEXT::ePerformanceWarning},
                         &DebugReportCallback});
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static auto SelectSwapchainProps(const vk::PhysicalDevice &physical_device,
                                 const vk::UniqueSurfaceKHR &surface) {
    // Get all capabilities
    const vk::SurfaceCapabilitiesKHR &surface_capas =
            physical_device.getSurfaceCapabilitiesKHR(surface.get());

    // Get the surface extent size
    VkExtent2D swapchain_extent = surface_capas.currentExtent;
    if (swapchain_extent.width == std::numeric_limits<uint32_t>::max()) {
        // If the surface size is undefined, setting screen size is requested.
        const uint32_t WIN_W = 256;
        const uint32_t WIN_H = 256;
        const auto &min_ex = surface_capas.minImageExtent;
        const auto &max_ex = surface_capas.maxImageExtent;
        swapchain_extent.width = Clamp(WIN_W, min_ex.width, max_ex.width);
        swapchain_extent.height = Clamp(WIN_H, min_ex.height, max_ex.height);
    }

    // Set swapchain pre-transform
    vk::SurfaceTransformFlagBitsKHR pre_trans = surface_capas.currentTransform;
    if (surface_capas.supportedTransforms &
        vk::SurfaceTransformFlagBitsKHR::eIdentity) {
        pre_trans = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    }

    // Set swapchain composite alpha
    vk::CompositeAlphaFlagBitsKHR composite_alpha;
    const auto &suppored_flag = surface_capas.supportedCompositeAlpha;
    if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
    } else if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
    } else if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::eInherit) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
    } else {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    }

    const uint32_t min_img_cnt = surface_capas.minImageCount;

    return std::make_tuple(swapchain_extent, pre_trans, composite_alpha,
                           min_img_cnt);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
vk::UniqueDeviceMemory AllocMemory(
        const vk::UniqueDevice &device,
        const vk::PhysicalDevice &physical_device,
        const vk::MemoryRequirements &memory_requs,
        const vk::MemoryPropertyFlags &require_flags) {
    auto actual_memory_props = physical_device.getMemoryProperties();
    uint32_t type_bits = memory_requs.memoryTypeBits;
    uint32_t type_idx = uint32_t(~0);
    for (uint32_t i = 0; i < actual_memory_props.memoryTypeCount; i++) {
        if ((type_bits & 1) &&
            IsSufficient(actual_memory_props.memoryTypes[i].propertyFlags,
                         require_flags)) {
            type_idx = i;
            break;
        }
        type_bits >>= 1;
    }
    if (type_idx == uint32_t(~0)) {
        throw std::runtime_error("Failed to allocate requested memory");
    }

    return device->allocateMemoryUnique({memory_requs.size, type_idx});
}

static vk::UniqueImageView CreateImageView(const vk::Image &img,
                                           const vk::Format &format,
                                           const vk::ImageAspectFlags &aspects,
                                           const vk::UniqueDevice &device) {
    const vk::ComponentMapping comp_mapping(
            vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
            vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);
    vk::ImageSubresourceRange sub_res_range(aspects, 0, 1, 0, 1);
    auto view = device->createImageViewUnique({vk::ImageViewCreateFlags(), img,
                                               vk::ImageViewType::e2D, format,
                                               comp_mapping, sub_res_range});
    return view;
}

static void SendToDevice(const vk::UniqueDevice &device,
                         const vk::UniqueDeviceMemory &dev_mem,
                         const vk::DeviceSize &dev_mem_size, const void *data,
                         uint64_t n_bytes) {
    uint8_t *dev_p = static_cast<uint8_t *>(
            device->mapMemory(dev_mem.get(), 0, dev_mem_size));
    memcpy(dev_p, data, n_bytes);
    device->unmapMemory(dev_mem.get());
}

static vk::UniqueSampler CreateSampler(const vk::UniqueDevice &device,
                                       const vk::Filter &mag_filter,
                                       const vk::Filter &min_filter,
                                       const vk::SamplerMipmapMode &mipmap,
                                       const vk::SamplerAddressMode &addr_u,
                                       const vk::SamplerAddressMode &addr_v,
                                       const vk::SamplerAddressMode &addr_w) {
    const float mip_lod_bias = 0.f;
    const bool anisotropy_enable = false;
    const float max_anisotropy = 16.f;
    const bool compare_enable = false;
    const vk::CompareOp compare_op = vk::CompareOp::eNever;
    const float min_lod = 0.f;
    const float max_lod = 0.f;
    const vk::BorderColor border_color = vk::BorderColor::eFloatOpaqueBlack;

    return device->createSamplerUnique(
            {vk::SamplerCreateFlags(), mag_filter, min_filter, mipmap, addr_u,
             addr_v, addr_w, mip_lod_bias, anisotropy_enable, max_anisotropy,
             compare_enable, compare_op, min_lod, max_lod, border_color});
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static void SetImageLayout(const vk::UniqueCommandBuffer &cmd_buf,
                           const ImagePackPtr &img_pack,
                           const vk::ImageLayout &new_img_layout) {
    // Check the need to set layout
    const vk::ImageLayout old_img_layout = img_pack->layout;
    if (old_img_layout == new_img_layout) {
        return;
    }
    // Shift layout variable
    img_pack->layout = new_img_layout;

    // Decide source access mask
    const vk::AccessFlags src_access_mask = [&]() -> vk::AccessFlags {
        switch (old_img_layout) {
            case vk::ImageLayout::eTransferDstOptimal:
                return vk::AccessFlagBits::eTransferWrite;
            case vk::ImageLayout::ePreinitialized:
                return vk::AccessFlagBits::eHostWrite;
            case vk::ImageLayout::eGeneral:
            case vk::ImageLayout::eUndefined: return {};  // empty
            default:
                throw std::runtime_error("Unexpected old image layout (A)");
        }
    }();
    // Decide source stage
    const vk::PipelineStageFlags src_stage = [&]() -> vk::PipelineStageFlags {
        switch (old_img_layout) {
            case vk::ImageLayout::eGeneral:
            case vk::ImageLayout::ePreinitialized:
                return vk::PipelineStageFlagBits::eHost;
            case vk::ImageLayout::eTransferDstOptimal:
                return vk::PipelineStageFlagBits::eTransfer;
            case vk::ImageLayout::eUndefined:
                return vk::PipelineStageFlagBits::eTopOfPipe;
            default:
                throw std::runtime_error("Unexpected old image layout (B)");
        }
    }();
    // Decide destination access mask
    const vk::AccessFlags dst_access_mask = [&]() -> vk::AccessFlags {
        switch (new_img_layout) {
            case vk::ImageLayout::eColorAttachmentOptimal:
                return vk::AccessFlagBits::eColorAttachmentWrite;
            case vk::ImageLayout::eDepthStencilAttachmentOptimal:
                return vk::AccessFlagBits::eDepthStencilAttachmentRead |
                       vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            case vk::ImageLayout::eGeneral: return {};  // empty
            case vk::ImageLayout::eShaderReadOnlyOptimal:
                return vk::AccessFlagBits::eShaderRead;
            case vk::ImageLayout::eTransferSrcOptimal:
                return vk::AccessFlagBits::eTransferRead;
            case vk::ImageLayout::eTransferDstOptimal:
                return vk::AccessFlagBits::eTransferWrite;
            default:
                throw std::runtime_error("Unexpected new image layout (A)");
        }
    }();
    // Decide destination stage
    const vk::PipelineStageFlags dst_stage = [&]() -> vk::PipelineStageFlags {
        switch (new_img_layout) {
            case vk::ImageLayout::eColorAttachmentOptimal:
                return vk::PipelineStageFlagBits::eColorAttachmentOutput;
            case vk::ImageLayout::eDepthStencilAttachmentOptimal:
                return vk::PipelineStageFlagBits::eEarlyFragmentTests;
            case vk::ImageLayout::eGeneral:
                return vk::PipelineStageFlagBits::eHost;
            case vk::ImageLayout::eShaderReadOnlyOptimal:
                return vk::PipelineStageFlagBits::eFragmentShader;
            case vk::ImageLayout::eTransferDstOptimal:
            case vk::ImageLayout::eTransferSrcOptimal:
                return vk::PipelineStageFlagBits::eTransfer;
            default:
                throw std::runtime_error("Unexpected new image layout (B)");
        }
    }();
    // Decide aspect mask
    const vk::ImageAspectFlags aspect_mask = [&]() -> vk::ImageAspectFlags {
        if (new_img_layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            const vk::Format &format = img_pack->format;
            if (format == vk::Format::eD32SfloatS8Uint ||
                format == vk::Format::eD24UnormS8Uint) {
                return vk::ImageAspectFlagBits::eDepth |
                       vk::ImageAspectFlagBits::eStencil;
            } else {
                return vk::ImageAspectFlagBits::eDepth;
            }
        } else {
            return vk::ImageAspectFlagBits::eColor;
        }
    }();

    // Set image layout
    vk::ImageSubresourceRange img_subresource_range(aspect_mask, 0, 1, 0, 1);
    vk::ImageMemoryBarrier img_memory_barrier(
            src_access_mask, dst_access_mask, old_img_layout, new_img_layout,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            img_pack->img.get(), img_subresource_range);
    return cmd_buf->pipelineBarrier(src_stage, dst_stage, {}, nullptr, nullptr,
                                    img_memory_barrier);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
auto PrepareFrameBuffer(const RenderPassPackPtr &render_pass_pack,
                        const std::vector<ImagePackPtr> &imgs,
                        const vk::Extent2D &size_org) {
    // Check the number of input images
    const auto &attachment_descs = render_pass_pack->attachment_descs;
    if (attachment_descs.size() != imgs.size() || imgs.empty()) {
        throw std::runtime_error("n_imgs != n_att_descs (FrameBuffer)");
    }

    // Extract size from the first image
    vk::Extent2D size = size_org;
    if (size.width == 0 || size.height == 0) {
        size = imgs[0]->size;
    }

    // Check image sizes
    for (uint32_t i = 0; i < imgs.size(); i++) {
        if (imgs[i] && imgs[i]->size != size) {
            throw std::runtime_error("Image sizes are not match (FrameBuffer)");
        }
    }

    // Create attachments
    std::vector<vk::ImageView> attachments;
    attachments.reserve(imgs.size());
    for (auto &&img : imgs) {
        if (img) {
            attachments.push_back(*img->view);
        } else {
            attachments.emplace_back();
        }
    }
    return std::make_tuple(size, std::move(attachments));
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

EShLanguage CvtShaderStage(const vk::ShaderStageFlagBits &stage) {
    switch (stage) {
        case vk::ShaderStageFlagBits::eVertex: return EShLangVertex;
        case vk::ShaderStageFlagBits::eTessellationControl:
            return EShLangTessControl;
        case vk::ShaderStageFlagBits::eTessellationEvaluation:
            return EShLangTessEvaluation;
        case vk::ShaderStageFlagBits::eGeometry: return EShLangGeometry;
        case vk::ShaderStageFlagBits::eFragment: return EShLangFragment;
        case vk::ShaderStageFlagBits::eCompute: return EShLangCompute;
        case vk::ShaderStageFlagBits::eRaygenNV: return EShLangRayGenNV;
        case vk::ShaderStageFlagBits::eAnyHitNV: return EShLangAnyHitNV;
        case vk::ShaderStageFlagBits::eClosestHitNV: return EShLangClosestHitNV;
        case vk::ShaderStageFlagBits::eMissNV: return EShLangMissNV;
        case vk::ShaderStageFlagBits::eIntersectionNV:
            return EShLangIntersectNV;
        case vk::ShaderStageFlagBits::eCallableNV: return EShLangCallableNV;
        case vk::ShaderStageFlagBits::eTaskNV: return EShLangTaskNV;
        case vk::ShaderStageFlagBits::eMeshNV: return EShLangMeshNV;
        default: throw std::runtime_error("Invalid shader type");
    }
}

std::vector<unsigned int> CompileGLSL(const vk::ShaderStageFlagBits &vk_stage,
                                      const std::string &source) {
    const EShLanguage stage = CvtShaderStage(vk_stage);
    const EShMessages rules =
            static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);

    // Set source string
    glslang::TShader shader(stage);
    const char *source_strs[1] = {source.data()};
    shader.setStrings(source_strs, 1);

    // Error handling function
    auto throw_shader_err = [&](const std::string &tag) {
        std::stringstream err_ss;
        err_ss << tag << std::endl;
        err_ss << shader.getInfoLog() << std::endl;
        err_ss << shader.getInfoDebugLog() << std::endl;
        err_ss << " Shader Source:" << std::endl;
        std::vector<std::string> source_lines = Split(source);
        for (size_t i = 0; i < source_lines.size(); i++) {
            err_ss << "  " << (i + 1) << ": " << source_lines[i] << std::endl;
        }
        throw std::runtime_error(err_ss.str());
    };

    // Parse GLSL with SPIR-V and Vulkan rules
    if (!shader.parse(&glslang::DefaultTBuiltInResource, 100, false, rules)) {
        throw_shader_err("Failed to parse GLSL");
    }

    // Link to program
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(rules)) {
        throw_shader_err("Failed to link GLSL");
    }

    // Convert GLSL to SPIRV
    std::vector<unsigned int> spv_data;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spv_data);
    return spv_data;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
}  // namespace

// -----------------------------------------------------------------------------
// -------------------------------- Info Prints --------------------------------
// -----------------------------------------------------------------------------

#if defined(__ANDROID__)
// Android print
void PrintInfo(const std::string &str) {
    __android_log_print(ANDROID_LOG_INFO, "VKW", "%s", str.c_str());
}
void PrintErr(const std::string &str) {
    __android_log_print(ANDROID_LOG_ERROR, "VKW", "%s", str.c_str());
}
#else
// Standard print
void PrintInfo(const std::string &str) {
    std::cout << str << std::endl;
}
void PrintErr(const std::string &str) {
    std::cerr << str << std::endl;
}
#endif

void PrintInstanceLayerProps() {
    // Create information string
    std::stringstream ss;
    ss << "* InstanceLayerProperties" << std::endl;
    const auto &props = vk::enumerateInstanceLayerProperties();
    for (uint32_t i = 0; i < props.size(); i++) {
        const auto prop = props[i];
        ss << "  " << i << ": " << prop.layerName
           << " (spec_version: " << prop.specVersion << ")"
           << " (impl_version: " << prop.implementationVersion << ")";
        ss << std::endl;
        ss << "    " << prop.description;
        ss << std::endl;
    }
    // Print
    PrintInfo(ss.str());
}

void PrintInstanceExtensionProps() {
    // Create information string
    std::stringstream ss;
    ss << "* InstanceExtensionProperties" << std::endl;
    const auto &props = vk::enumerateInstanceExtensionProperties();
    for (uint32_t i = 0; i < props.size(); i++) {
        const auto prop = props[i];
        ss << "  " << i << ": " << prop.extensionName
           << " (spec_version: " << prop.specVersion << ")";
        ss << std::endl;
    }
    // Print
    PrintInfo(ss.str());
}

void PrintQueueFamilyProps(const vk::PhysicalDevice &physical_device) {
    // Create information string
    std::stringstream ss;
    ss << "* QueueFamilyProperties" << std::endl;
    const auto &props = physical_device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < props.size(); i++) {
        const auto prop = props[i];
        const auto &flags_str = vk::to_string(prop.queueFlags);
        const auto &max_queue_cnt = prop.queueCount;
        ss << "  " << i << ": " << flags_str
           << "  (max_queue_cnt:" << max_queue_cnt << ")" << std::endl;
    }
    // Print
    PrintInfo(ss.str());
}

// -----------------------------------------------------------------------------
// -------------------------------- FPS counter --------------------------------
// -----------------------------------------------------------------------------
void DefaultFpsFunc(float fps) {
    std::stringstream ss;
    ss << "Fps: " << fps;
    PrintInfo(ss.str());
}

void PrintFps(std::function<void(float)> print_func, int show_interval) {
    static int s_count = -2;  // Some starting offset
    static auto s_start_clk = std::chrono::system_clock::now();

    // Count up
    s_count++;
    // Print per interval
    if (s_count % show_interval == show_interval - 1) {
        // Compute fps
        auto end_clk = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end_clk - s_start_clk;
        const float fps = static_cast<float>(show_interval) /
                          static_cast<float>(elapsed.count());
        // Print
        print_func(fps);
        // Shift clock
        s_count = 0;
        s_start_clk = end_clk;
    }
}

// -----------------------------------------------------------------------------
// ----------------------------------- Window ----------------------------------
// -----------------------------------------------------------------------------
#if defined(__ANDROID__)
// ------------------------- ANativeWindow for Android -------------------------
static void WindowDeleter(ANativeWindow *ptr) {
    ANativeWindow_release(ptr);
}

WindowPtr InitANativeWindow(JNIEnv *jenv, jobject jsurface) {
    ANativeWindow *ptr = ANativeWindow_fromSurface(jenv, jsurface);
    return WindowPtr(ptr, WindowDeleter);
}

#else
// --------------------------- GLFWWindow for Desktop --------------------------
static void WindowDeleter(GLFWwindow *ptr) {
    glfwDestroyWindow(ptr);
}

WindowPtr InitGLFWWindow(const std::string &win_name, uint32_t win_w,
                         uint32_t win_h) {
    // Initialize GLFW
    static bool s_is_inited = false;
    if (!s_is_inited) {
        s_is_inited = true;
        glfwInit();
        atexit([]() { glfwTerminate(); });
    }

    // Create GLFW window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *ptr =
            glfwCreateWindow(static_cast<int>(win_w), static_cast<int>(win_h),
                             win_name.c_str(), nullptr, nullptr);
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("No Vulkan support");
    }
    return WindowPtr(ptr, WindowDeleter);
}
#endif

// -----------------------------------------------------------------------------
// --------------------------------- Instance ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueInstance CreateInstance(const std::string &app_name,
                                  uint32_t app_version,
                                  const std::string &engine_name,
                                  uint32_t engine_version, bool debug_enable,
                                  bool surface_enable) {
    // Initialize dispatcher with `vkGetInstanceProcAddr`, to get the instance
    // independent function pointers
    PFN_vkGetInstanceProcAddr get_vk_instance_proc_addr =
            vk::DynamicLoader()
                    .template getProcAddress<PFN_vkGetInstanceProcAddr>(
                            "vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(get_vk_instance_proc_addr);

    // Decide Vulkan layer and extensions
    const auto &enabled_layer = GetEnabledLayers(debug_enable);
    const auto &enabled_exts = GetEnabledExts(debug_enable, surface_enable);

    // Create instance
    vk::ApplicationInfo app_info = {app_name.c_str(), app_version,
                                    engine_name.c_str(), engine_version,
                                    VK_API_VERSION_1_1};
    vk::UniqueInstance instance = vk::createInstanceUnique(
            {vk::InstanceCreateFlags(), &app_info,
             static_cast<uint32_t>(enabled_layer.size()),
             DataSafety(enabled_layer),
             static_cast<uint32_t>(enabled_exts.size()),
             DataSafety(enabled_exts)});

    // Initialize dispatcher with Instance to get all the other function ptrs.
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    // Create debug messenger or debug report
    if (debug_enable) {
        RegisterDebugCalback(instance);
    }

    return instance;
}

// -----------------------------------------------------------------------------
// ------------------------------ PhysicalDevice -------------------------------
// -----------------------------------------------------------------------------
std::vector<vk::PhysicalDevice> GetPhysicalDevices(
        const vk::UniqueInstance &instance) {
    return instance->enumeratePhysicalDevices();
}

vk::PhysicalDevice GetFirstPhysicalDevice(const vk::UniqueInstance &instance) {
    // Get physical devices
    auto physical_devices = GetPhysicalDevices(instance);
    // Select the first one
    const size_t n_phy_devices = physical_devices.size();
    if (n_phy_devices == 0) {
        throw std::runtime_error("No physical devices");
    }
    if (1 < n_phy_devices) {
        std::stringstream ss;
        ss << "Non single physical deivces (" << n_phy_devices << "), ";
        ss << "Using first one.";
        vkw::PrintInfo(ss.str());
    }
    return physical_devices.front();
}

// -----------------------------------------------------------------------------
// ---------------------------------- Surface ----------------------------------
// -----------------------------------------------------------------------------
#if defined(__ANDROID__)
// Android version
vk::UniqueSurfaceKHR CreateSurface(const vk::UniqueInstance &instance,
                                   const WindowPtr &window) {
    // Create Android surface
    return instance->createAndroidSurfaceKHRUnique(
            {vk::AndroidSurfaceCreateFlagsKHR(), window.get()});
}

#else
// Desktop version
vk::UniqueSurfaceKHR CreateSurface(const vk::UniqueInstance &instance,
                                   const WindowPtr &window) {
    // Create a window surface (GLFW)
    VkSurfaceKHR s_raw = nullptr;
    VkResult err =
            glfwCreateWindowSurface(*instance, window.get(), nullptr, &s_raw);
    if (err) {
        throw std::runtime_error("Failed to create window surface");
    }
    // Wrap with smart handler
    using Deleter =
            vk::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>;
    Deleter deleter(*instance, nullptr, VULKAN_HPP_DEFAULT_DISPATCHER);
    return vk::UniqueSurfaceKHR(s_raw, deleter);
}
#endif

vk::Format GetSurfaceFormat(const vk::PhysicalDevice &physical_device,
                            const vk::UniqueSurfaceKHR &surface) {
    auto surface_formats = physical_device.getSurfaceFormatsKHR(surface.get());
    assert(!surface_formats.empty());
    vk::Format surface_format = surface_formats[0].format;
    if (surface_format == vk::Format::eUndefined) {
        surface_format = vk::Format::eB8G8R8A8Unorm;
    }
    return surface_format;
}

// -----------------------------------------------------------------------------
// -------------------------------- Queue Family -------------------------------
// -----------------------------------------------------------------------------
std::vector<uint32_t> GetQueueFamilyIdxs(
        const vk::PhysicalDevice &physical_device,
        const vk::QueueFlags &queue_flags) {
    const auto &props = physical_device.getQueueFamilyProperties();

    // Search sufficient queue family indices
    std::vector<uint32_t> queue_family_idxs;
    for (uint32_t i = 0; i < props.size(); i++) {
        if (IsSufficient(props[i].queueFlags, queue_flags)) {
            queue_family_idxs.push_back(i);
        }
    }
    return queue_family_idxs;
}

uint32_t GetGraphicPresentQueueFamilyIdx(
        const vk::PhysicalDevice &physical_device,
        const vk::UniqueSurfaceKHR &surface,
        const vk::QueueFlags &queue_flags) {
    // Get graphics queue family indices
    auto graphic_idxs = GetQueueFamilyIdxs(physical_device, queue_flags);

    // Extract present queue indices
    std::vector<uint32_t> graphic_present_idxs;
    for (auto &&idx : graphic_idxs) {
        if (physical_device.getSurfaceSupportKHR(idx, surface.get())) {
            graphic_present_idxs.push_back(idx);
        }
    }

    // Check status
    if (graphic_present_idxs.size() == 0) {
        throw std::runtime_error("No sufficient queue for graphic present");
    }
    // Return the first index
    return graphic_present_idxs.front();
}

// -----------------------------------------------------------------------------
// ----------------------------------- Device ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueDevice CreateDevice(uint32_t queue_family_idx,
                              const vk::PhysicalDevice &physical_device,
                              uint32_t n_queues, bool swapchain_support) {
    // Create queue create info
    std::vector<float> queue_priorites(n_queues, 0.f);
    vk::DeviceQueueCreateInfo device_queue_create_info = {
            vk::DeviceQueueCreateFlags(), queue_family_idx, n_queues,
            queue_priorites.data()};

    // Create device extension strings
    std::vector<const char *> device_exts;
    if (swapchain_support) {
        device_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    const uint32_t n_device_exts = static_cast<uint32_t>(device_exts.size());

    // Create a logical device
    vk::UniqueDevice device = physical_device.createDeviceUnique(
            {vk::DeviceCreateFlags(), 1, &device_queue_create_info, 0, nullptr,
             n_device_exts, DataSafety(device_exts)});

    // Initialize dispatcher for device
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    return device;
}

// -----------------------------------------------------------------------------
// ------------------------------- Asynchronous --------------------------------
// -----------------------------------------------------------------------------
FencePtr CreateFence(const vk::UniqueDevice &device) {
    auto fence = device->createFenceUnique({});
    return FencePtr(new vk::UniqueFence(std::move(fence)));
}

void ResetFence(const vk::UniqueDevice &device, const FencePtr &fence) {
    device->resetFences(1, &fence->get());
}

vk::Result WaitForFence(const vk::UniqueDevice &device, const FencePtr &fence,
                        uint64_t timeout) {
    // Wait during `timeout` nano-seconds
    return device->waitForFences(1, &fence->get(), false, timeout);
}

vk::Result WaitForFences(const vk::UniqueDevice &device,
                         const std::vector<FencePtr> &fences, bool wait_all,
                         uint64_t timeout) {
    // Repack fences
    const uint32_t n_fences = static_cast<uint32_t>(fences.size());
    std::vector<vk::Fence> fences_raw;
    fences_raw.reserve(n_fences);
    for (auto &&f : fences) {
        fences_raw.push_back(f->get());
    }

    // Wait during `timeout` nano-seconds
    return device->waitForFences(n_fences, fences_raw.data(), wait_all,
                                 timeout);
}

EventPtr CreateEvent(const vk::UniqueDevice &device) {
    auto event = device->createEventUnique({});
    return EventPtr(new vk::UniqueEvent{std::move(event)});
}

SemaphorePtr CreateSemaphore(const vk::UniqueDevice &device) {
    auto semaphore = device->createSemaphoreUnique({});
    return SemaphorePtr(new vk::UniqueSemaphore{std::move(semaphore)});
}

// -----------------------------------------------------------------------------
// --------------------------------- Swapchain ---------------------------------
// -----------------------------------------------------------------------------
SwapchainPackPtr CreateSwapchainPack(const vk::PhysicalDevice &physical_device,
                                     const vk::UniqueDevice &device,
                                     const vk::UniqueSurfaceKHR &surface,
                                     const vk::Format &surface_format_raw,
                                     const vk::ImageUsageFlags &usage) {
    // Set swapchain present mode
    const vk::PresentModeKHR swapchain_present_mode = vk::PresentModeKHR::eFifo;

    // Get the supported surface VkFormats
    auto surface_format = (surface_format_raw == vk::Format::eUndefined) ?
                                  GetSurfaceFormat(physical_device, surface) :
                                  surface_format_raw;

    // Select properties from capabilities
    auto props = SelectSwapchainProps(physical_device, surface);
    const auto &swapchain_extent = std::get<0>(props);
    const auto &pre_trans = std::get<1>(props);
    const auto &composite_alpha = std::get<2>(props);
    const auto &min_img_cnt = std::get<3>(props);

    // Create swapchain
    vk::UniqueSwapchainKHR swapchain = device->createSwapchainKHRUnique(
            {vk::SwapchainCreateFlagsKHR(), surface.get(), min_img_cnt,
             surface_format, vk::ColorSpaceKHR::eSrgbNonlinear,
             swapchain_extent, 1, usage, vk::SharingMode::eExclusive, 0,
             nullptr, pre_trans, composite_alpha, swapchain_present_mode, true,
             nullptr});

    // Create image views
    auto swapchain_imgs = device->getSwapchainImagesKHR(swapchain.get());
    std::vector<vk::UniqueImageView> img_views;
    img_views.reserve(swapchain_imgs.size());
    for (auto img : swapchain_imgs) {
        auto img_view = CreateImageView(
                img, surface_format, vk::ImageAspectFlagBits::eColor, device);
        img_views.push_back(std::move(img_view));
    }

    return SwapchainPackPtr(new SwapchainPack{
            std::move(swapchain), std::move(img_views), swapchain_extent});
}

vk::Result AcquireNextImage(uint32_t *out_img_idx,
                            const vk::UniqueDevice &device,
                            const SwapchainPackPtr &swapchain_pack,
                            const SemaphorePtr &signal_semaphore,
                            const FencePtr &signal_fence, uint64_t timeout) {
    // Escape nullptr
    vk::Semaphore semaphore_raw =
            signal_semaphore ? signal_semaphore->get() : vk::Semaphore();
    vk::Fence fence_raw = signal_fence ? signal_fence->get() : vk::Fence();
    // Acquire
    return device->acquireNextImageKHR(swapchain_pack->swapchain.get(), timeout,
                                       semaphore_raw, fence_raw, out_img_idx);
}

// -----------------------------------------------------------------------------
// ----------------------------------- Buffer ----------------------------------
// -----------------------------------------------------------------------------
BufferPackPtr CreateBufferPack(const vk::PhysicalDevice &physical_device,
                               const vk::UniqueDevice &device,
                               const vk::DeviceSize &size,
                               const vk::BufferUsageFlags &usage,
                               const vk::MemoryPropertyFlags &memory_props) {
    // Create buffer
    auto buf =
            device->createBufferUnique({vk::BufferCreateFlags(), size, usage});

    // Allocate memory
    auto memory_requs = device->getBufferMemoryRequirements(*buf);
    auto device_mem =
            AllocMemory(device, physical_device, memory_requs, memory_props);

    // Bind memory
    device->bindBufferMemory(buf.get(), device_mem.get(), 0);

    return BufferPackPtr(new BufferPack{
            std::move(buf), size, std::move(device_mem), memory_requs.size});
}

void SendToDevice(const vk::UniqueDevice &device, const BufferPackPtr &buf_pack,
                  const void *data, uint64_t n_bytes) {
    SendToDevice(device, buf_pack->dev_mem, buf_pack->dev_mem_size, data,
                 n_bytes);
}

// -----------------------------------------------------------------------------
// ----------------------------------- Image -----------------------------------
// -----------------------------------------------------------------------------
ImagePackPtr CreateImagePack(const vk::PhysicalDevice &physical_device,
                             const vk::UniqueDevice &device,
                             const vk::Format &format, const vk::Extent2D &size,
                             const vk::ImageUsageFlags &usage_org,
                             const vk::ImageAspectFlags &aspects,
                             bool is_staging, bool is_shared) {
    // Decide modes about staging
    auto modes = [&]() {
        if (is_staging) {
            return std::make_tuple(
                    vk::ImageTiling::eOptimal, vk::ImageLayout::eUndefined,
                    vk::MemoryPropertyFlags{},
                    (usage_org | vk::ImageUsageFlagBits::eTransferDst));
        } else {
            return std::make_tuple(
                    vk::ImageTiling::eLinear, vk::ImageLayout::ePreinitialized,
                    vk::MemoryPropertyFlagBits::eHostCoherent |
                            vk::MemoryPropertyFlagBits::eHostVisible,
                    usage_org);
        }
    }();
    const vk::ImageTiling &tiling = std::get<0>(modes);
    const vk::ImageLayout &initial_layout = std::get<1>(modes);
    const vk::MemoryPropertyFlags &memory_props = std::get<2>(modes);
    const vk::ImageUsageFlags &usage = std::get<3>(modes);

    // Select sharing mode
    const vk::SharingMode sharing = is_shared ? vk::SharingMode::eConcurrent :
                                                vk::SharingMode::eExclusive;

    // Create image
    auto img = device->createImageUnique(
            {vk::ImageCreateFlags(), vk::ImageType::e2D, format,
             vk::Extent3D(size, 1), 1, 1, vk::SampleCountFlagBits::e1, tiling,
             usage, sharing, 0, nullptr, initial_layout});

    // Allocate memory
    auto memory_requs = device->getImageMemoryRequirements(*img);
    auto device_mem =
            AllocMemory(device, physical_device, memory_requs, memory_props);
    auto dev_mem_size = memory_requs.size;

    // Bind memory
    device->bindImageMemory(img.get(), device_mem.get(), 0);

    // Create image view
    auto img_view = CreateImageView(*img, format, aspects, device);

    // Create transfer buffer for only staging mode
    BufferPackPtr trans_buf_pack;
    if (is_staging) {
        trans_buf_pack = CreateBufferPack(
                physical_device, device, dev_mem_size,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostCoherent |
                        vk::MemoryPropertyFlagBits::eHostVisible);
    }

    // Construct image pack
    return ImagePackPtr(new ImagePack{std::move(img), std::move(img_view),
                                      format, size, std::move(device_mem),
                                      dev_mem_size, is_staging, initial_layout,
                                      std::move(trans_buf_pack)});
}

void SendToDevice(const vk::UniqueDevice &device, const ImagePackPtr &img_pack,
                  const void *data, uint64_t n_bytes,
                  const vk::UniqueCommandBuffer &cmd_buf) {
    if (img_pack->is_staging) {
        // Check command buffer existence
        if (!cmd_buf.get()) {
            throw std::runtime_error("No command buffer to send staging image");
        }

        // Send to buffer
        SendToDevice(device, img_pack->trans_buf_pack, data, n_bytes);

        // Transfer from buffer to image
        SetImageLayout(cmd_buf, img_pack, vk::ImageLayout::eTransferDstOptimal);
        const auto &extent = img_pack->size;
        vk::BufferImageCopy copy_region(
                0, extent.width, extent.height,
                vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0,
                                           0, 1),
                vk::Offset3D(0, 0, 0), vk::Extent3D(extent, 1));
        cmd_buf->copyBufferToImage(
                img_pack->trans_buf_pack->buf.get(), img_pack->img.get(),
                vk::ImageLayout::eTransferDstOptimal, copy_region);
    } else {
        // Send to device directly
        SendToDevice(device, img_pack->dev_mem, img_pack->dev_mem_size, data,
                     n_bytes);
    }

    // Set layout for shader read
    SetImageLayout(cmd_buf, img_pack, vk::ImageLayout::eShaderReadOnlyOptimal);
}

// -----------------------------------------------------------------------------
// ---------------------------------- Texture ----------------------------------
// -----------------------------------------------------------------------------
TexturePackPtr CreateTexturePack(const ImagePackPtr &img_pack,
                                 const vk::UniqueDevice &device,
                                 const vk::Filter &mag_filter,
                                 const vk::Filter &min_filter,
                                 const vk::SamplerMipmapMode &mipmap,
                                 const vk::SamplerAddressMode &addr_u,
                                 const vk::SamplerAddressMode &addr_v,
                                 const vk::SamplerAddressMode &addr_w) {
    // Create sampler
    auto sampler = CreateSampler(device, mag_filter, min_filter, mipmap, addr_u,
                                 addr_v, addr_w);

    // Construct texture pack
    return TexturePackPtr(new TexturePack{img_pack, std::move(sampler)});
}

void SendToDevice(const vk::UniqueDevice &device,
                  const TexturePackPtr &tex_pack, const void *data,
                  uint64_t n_bytes, const vk::UniqueCommandBuffer &cmd_buf) {
    SendToDevice(device, tex_pack->img_pack, data, n_bytes, cmd_buf);
}

// -----------------------------------------------------------------------------
// ------------------------------- DescriptorSet -------------------------------
// -----------------------------------------------------------------------------
DescSetPackPtr CreateDescriptorSetPack(const vk::UniqueDevice &device,
                                       const std::vector<DescSetInfo> &infos) {
    const uint32_t n_bindings = static_cast<uint32_t>(infos.size());

    // Parse into raw array of bindings, pool sizes
    std::vector<vk::DescriptorSetLayoutBinding> bindings_raw;
    std::vector<vk::DescriptorPoolSize> poolsizes_raw;
    bindings_raw.reserve(n_bindings);
    poolsizes_raw.reserve(n_bindings);
    uint32_t desc_cnt_sum = 0;
    for (uint32_t i = 0; i < n_bindings; i++) {
        // Fetch from tuple
        const vk::DescriptorType &desc_type = std::get<0>(infos[i]);
        const uint32_t &desc_cnt = std::get<1>(infos[i]);
        const vk::ShaderStageFlags &shader_stage = std::get<2>(infos[i]);
        // Sum up descriptor count
        desc_cnt_sum += desc_cnt;
        // Push to bindings
        bindings_raw.emplace_back(i, desc_type, desc_cnt, shader_stage);
        // Push to pool sizes
        poolsizes_raw.emplace_back(desc_type, desc_cnt);
    }

    // Create DescriptorSetLayout
    auto desc_set_layout = device->createDescriptorSetLayoutUnique(
            {vk::DescriptorSetLayoutCreateFlags(), n_bindings,
             DataSafety(bindings_raw)});
    // Create DescriptorPool
    auto desc_pool = device->createDescriptorPoolUnique(
            {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, desc_cnt_sum,
             n_bindings, DataSafety(poolsizes_raw)});
    // Create DescriptorSet
    auto desc_sets = device->allocateDescriptorSetsUnique(
            {*desc_pool, 1, &*desc_set_layout});
    auto &desc_set = desc_sets[0];

    return DescSetPackPtr(new DescSetPack{std::move(desc_set_layout),
                                          std::move(desc_pool),
                                          std::move(desc_set), infos});
}

WriteDescSetPackPtr CreateWriteDescSetPack() {
    return std::make_shared<WriteDescSetPack>();
}

void AddWriteDescSet(WriteDescSetPackPtr &write_pack,
                     const DescSetPackPtr &desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<BufferPackPtr> &buf_packs) {
    // Fetch form and check with DescSetInfo
    const DescSetInfo &desc_set_info =
            desc_set_pack->desc_set_info[binding_idx];
    const vk::DescriptorType desc_type = std::get<0>(desc_set_info);
    const uint32_t desc_cnt = std::get<1>(desc_set_info);
    if (desc_cnt != static_cast<uint32_t>(buf_packs.size())) {
        throw std::runtime_error("Invalid descriptor count to write buffers");
    }

    // Create vector of DescriptorBufferInfo in the result pack
    auto &buf_infos = EmplaceBackEmpty(write_pack->desc_buf_info_vecs);
    // Create DescriptorBufferInfo
    for (auto &&buf_pack : buf_packs) {
        buf_infos.emplace_back(*buf_pack->buf, 0, VK_WHOLE_SIZE);
    }

    // Create and Add WriteDescriptorSet
    write_pack->write_desc_sets.emplace_back(
            *desc_set_pack->desc_set, binding_idx, 0, desc_cnt, desc_type,
            nullptr, buf_infos.data(), nullptr);  // TODO: buffer view
}

void AddWriteDescSet(WriteDescSetPackPtr &write_pack,
                     const DescSetPackPtr &desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<TexturePackPtr> &tex_packs) {
    // Fetch form and check with DescSetInfo
    const DescSetInfo &desc_set_info =
            desc_set_pack->desc_set_info[binding_idx];
    const vk::DescriptorType desc_type = std::get<0>(desc_set_info);
    const uint32_t desc_cnt = std::get<1>(desc_set_info);
    if (desc_cnt != static_cast<uint32_t>(tex_packs.size())) {
        throw std::runtime_error("Invalid descriptor count to write images");
    }
    // Note: desc_type should be `vk::DescriptorType::eCombinedImageSampler`

    // Create vector of DescriptorImageInfo in the result pack
    auto &img_infos = EmplaceBackEmpty(write_pack->desc_img_info_vecs);
    // Create DescriptorImageInfo
    for (auto &&tex_pack : tex_packs) {
        img_infos.emplace_back(*tex_pack->sampler, *tex_pack->img_pack->view,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    // Create and Add WriteDescriptorSet
    write_pack->write_desc_sets.emplace_back(
            *desc_set_pack->desc_set, binding_idx, 0, desc_cnt, desc_type,
            img_infos.data(), nullptr, nullptr);
}

void AddWriteDescSet(WriteDescSetPackPtr &write_pack,
                     const DescSetPackPtr &desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<ImagePackPtr> &img_packs) {
    // Fetch form and check with DescSetInfo
    const DescSetInfo &desc_set_info =
            desc_set_pack->desc_set_info[binding_idx];
    const vk::DescriptorType desc_type = std::get<0>(desc_set_info);
    const uint32_t desc_cnt = std::get<1>(desc_set_info);
    if (desc_cnt != static_cast<uint32_t>(img_packs.size())) {
        throw std::runtime_error("Invalid descriptor count to write images");
    }
    // Note: desc_type should be `vk::DescriptorType::eCombinedImageSampler`

    // Create vector of DescriptorImageInfo in the result pack
    auto &img_infos = EmplaceBackEmpty(write_pack->desc_img_info_vecs);
    // Create DescriptorImageInfo
    for (auto &&img_pack : img_packs) {
        const vk::Sampler empty_sampler = nullptr;
        img_infos.emplace_back(empty_sampler, *img_pack->view,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    // Create and Add WriteDescriptorSet
    write_pack->write_desc_sets.emplace_back(
            *desc_set_pack->desc_set, binding_idx, 0, desc_cnt, desc_type,
            img_infos.data(), nullptr, nullptr);
}

void UpdateDescriptorSets(const vk::UniqueDevice &device,
                          const WriteDescSetPackPtr &write_desc_set_pack) {
    device->updateDescriptorSets(write_desc_set_pack->write_desc_sets, nullptr);
}

// -----------------------------------------------------------------------------
// -------------------------------- RenderPass ---------------------------------
// -----------------------------------------------------------------------------
RenderPassPackPtr CreateRenderPassPack() {
    return std::make_shared<RenderPassPack>();
}

void AddAttachientDesc(RenderPassPackPtr &render_pass_pack,
                       const vk::Format &format,
                       const vk::AttachmentLoadOp &load_op,
                       const vk::AttachmentStoreOp &store_op,
                       const vk::ImageLayout &final_layout) {
    const auto sample_cnt = vk::SampleCountFlagBits::e1;
    const auto stencil_load_op = vk::AttachmentLoadOp::eDontCare;
    const auto stencil_store_op = vk::AttachmentStoreOp::eDontCare;
    const auto initial_layout = vk::ImageLayout::eUndefined;

    // Add attachment description
    render_pass_pack->attachment_descs.emplace_back(
            vk::AttachmentDescriptionFlags(), format, sample_cnt, load_op,
            store_op, stencil_load_op, stencil_store_op, initial_layout,
            final_layout);
}

void AddSubpassDesc(RenderPassPackPtr &render_pass_pack,
                    const std::vector<AttachmentRefInfo> &inp_attach_ref_infos,
                    const std::vector<AttachmentRefInfo> &col_attach_ref_infos,
                    const AttachmentRefInfo &depth_stencil_attach_ref_info) {
    // Allocate reference vectors
    auto &attachment_ref_vecs = render_pass_pack->attachment_ref_vecs;
    attachment_ref_vecs.resize(attachment_ref_vecs.size() + 3);

    // Add attachment references for inputs
    auto &inp_refs = attachment_ref_vecs.end()[-3];
    for (auto &&info : inp_attach_ref_infos) {
        inp_refs.emplace_back(std::get<0>(info), std::get<1>(info));
    }

    // Add attachment references for color outputs
    auto &col_refs = attachment_ref_vecs.end()[-2];
    for (auto &&info : col_attach_ref_infos) {
        col_refs.emplace_back(std::get<0>(info), std::get<1>(info));
    }

    // Add an attachment reference for depth-stencil
    auto &dep_refs = attachment_ref_vecs.end()[-1];
    auto &dep_info = depth_stencil_attach_ref_info;
    const bool depth_empty = (std::get<0>(dep_info) == uint32_t(~0));
    if (!depth_empty) {
        dep_refs.emplace_back(std::get<0>(dep_info), std::get<1>(dep_info));
    }

    // Collect attachment sizes and pointers
    const uint32_t n_inp = static_cast<uint32_t>(inp_refs.size());
    const uint32_t n_col = static_cast<uint32_t>(col_refs.size());
    // Unused options
    const vk::AttachmentReference *resolve_ref_data = nullptr;
    const uint32_t n_preserve_attachment = 0;
    const uint32_t *preserve_attachments_p = nullptr;

    // Add subpass description
    render_pass_pack->subpass_descs.emplace_back(
            vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics,
            n_inp, DataSafety(inp_refs), n_col, DataSafety(col_refs),
            resolve_ref_data, DataSafety(dep_refs), n_preserve_attachment,
            preserve_attachments_p);
}

void AddSubpassDepend(RenderPassPackPtr& render_pass_pack,
                      const DependInfo& src_depend,
                      const DependInfo& dst_depend,
                      const vk::DependencyFlags& depend_flags) {
    // Add subpass dependency info
    render_pass_pack->subpass_depends.emplace_back(
            src_depend.subpass_idx, dst_depend.subpass_idx,
            src_depend.stage_mask, dst_depend.stage_mask,
            src_depend.access_mask, dst_depend.access_mask,
            depend_flags);
}

void UpdateRenderPass(const vk::UniqueDevice &device,
                      RenderPassPackPtr &render_pass_pack) {
    const auto &att_descs = render_pass_pack->attachment_descs;
    const uint32_t n_att_descs = static_cast<uint32_t>(att_descs.size());
    const auto &sub_descs = render_pass_pack->subpass_descs;
    const uint32_t n_sub_descs = static_cast<uint32_t>(sub_descs.size());
    const auto &sub_depends = render_pass_pack->subpass_depends;
    const uint32_t n_sub_depends = static_cast<uint32_t>(sub_depends.size());

    // Create render pass instance
    render_pass_pack->render_pass = device->createRenderPassUnique(
            {vk::RenderPassCreateFlags(), n_att_descs, DataSafety(att_descs),
             n_sub_descs, DataSafety(sub_descs), n_sub_depends,
             DataSafety(sub_depends)});
}

// -----------------------------------------------------------------------------
// -------------------------------- FrameBuffer --------------------------------
// -----------------------------------------------------------------------------
FrameBufferPackPtr CreateFrameBuffer(const vk::UniqueDevice &device,
                                     const RenderPassPackPtr &render_pass_pack,
                                     const std::vector<ImagePackPtr> &imgs,
                                     const vk::Extent2D &size_org) {
    // Prepare frame buffer creation
    auto info = PrepareFrameBuffer(render_pass_pack, imgs, size_org);
    const vk::Extent2D &size = std::get<0>(info);
    const std::vector<vk::ImageView> &attachments = std::get<1>(info);
    const uint32_t n_layers = 1;

    // Create Frame Buffer
    auto frame_buffer = device->createFramebufferUnique(
            {vk::FramebufferCreateFlags(), *render_pass_pack->render_pass,
             static_cast<uint32_t>(attachments.size()), DataSafety(attachments),
             size.width, size.height, n_layers});

    return FrameBufferPackPtr(new FrameBufferPack{
            std::move(frame_buffer), size.width, size.height, n_layers});
}

std::vector<FrameBufferPackPtr> CreateFrameBuffers(
        const vk::UniqueDevice &device,
        const RenderPassPackPtr &render_pass_pack,
        const std::vector<ImagePackPtr> &imgs,
        const SwapchainPackPtr &swapchain) {
    // Prepare frame buffer creation
    auto info = PrepareFrameBuffer(render_pass_pack, imgs, swapchain->size);
    const vk::Extent2D &size = std::get<0>(info);
    std::vector<vk::ImageView> &attachments = std::get<1>(info);
    const uint32_t n_layers = 1;

    // Find attachment index
    uint32_t attach_idx = uint32_t(~0);
    for (size_t i = 0; i < imgs.size(); i++) {
        if (!imgs[i]) {
            attach_idx = static_cast<uint32_t>(i);
            break;
        }
    }
    if (attach_idx == uint32_t(~0)) {
        throw std::runtime_error("No attachment position for swapchain view");
    }

    // Create Frame Buffers
    std::vector<FrameBufferPackPtr> rets;
    rets.reserve(swapchain->views.size());
    for (auto &&view : swapchain->views) {
        // Overwrite swapchain image view
        attachments[attach_idx] = *view;
        // Create one Frame Buffer
        auto frame_buffer = device->createFramebufferUnique(
                {vk::FramebufferCreateFlags(), *render_pass_pack->render_pass,
                 static_cast<uint32_t>(attachments.size()),
                 DataSafety(attachments), size.width, size.height, n_layers});
        // Register
        rets.emplace_back(new FrameBufferPack{
                std::move(frame_buffer), size.width, size.height, n_layers});
    }

    return rets;
}

// -----------------------------------------------------------------------------
// -------------------------------- ShaderModule -------------------------------
// -----------------------------------------------------------------------------
GLSLCompiler::GLSLCompiler() {
    glslang::InitializeProcess();
}

GLSLCompiler::~GLSLCompiler() {
    glslang::FinalizeProcess();
}

ShaderModulePackPtr GLSLCompiler::compileFromString(
        const vk::UniqueDevice &device, const std::string &source,
        const vk::ShaderStageFlagBits &stage) {
    // Compile GLSL to SPIRV
    const std::vector<unsigned int> &spv_data = CompileGLSL(stage, source);
    // Create shader module
    auto shader_module = device->createShaderModuleUnique(
            {vk::ShaderModuleCreateFlags(),
             spv_data.size() * sizeof(unsigned int), spv_data.data()});
    return ShaderModulePackPtr(
            new ShaderModulePack{std::move(shader_module), stage});
}

// -----------------------------------------------------------------------------
// ---------------------------------- Pipeline ---------------------------------
// -----------------------------------------------------------------------------
PipelinePackPtr CreatePipeline(
        const vk::UniqueDevice &device,
        const std::vector<ShaderModulePackPtr> &shader_modules,
        const std::vector<VtxInputBindingInfo> &vtx_inp_binding_info,
        const std::vector<VtxInputAttribInfo> &vtx_inp_attrib_info,
        const PipelineInfo &pipeline_info,
        const std::vector<DescSetPackPtr> &desc_set_packs,
        const RenderPassPackPtr &render_pass_pack, uint32_t subpass_idx) {
    // Shader stage create infos
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stage_cis;
    shader_stage_cis.reserve(shader_modules.size());
    for (auto &&shader : shader_modules) {
        shader_stage_cis.emplace_back(vk::PipelineShaderStageCreateFlags(),
                                      shader->stage,
                                      shader->shader_module.get(), "main");
    }

    // Parse vertex input binding description
    std::vector<vk::VertexInputBindingDescription> vtx_inp_binding_descs;
    vtx_inp_binding_descs.reserve(vtx_inp_binding_info.size());
    for (auto &&info : vtx_inp_binding_info) {
        vtx_inp_binding_descs.emplace_back(info.binding_idx, info.stride,
                                           info.input_rate);
    }
    // Parse vertex input attribute description
    std::vector<vk::VertexInputAttributeDescription> vtx_inp_attrib_descs;
    vtx_inp_attrib_descs.reserve(vtx_inp_attrib_info.size());
    for (auto &&info : vtx_inp_attrib_info) {
        vtx_inp_attrib_descs.emplace_back(info.location, info.binding_idx,
                                          info.format, info.offset);
    }
    // Vertex input state create info
    vk::PipelineVertexInputStateCreateInfo vtx_inp_state_ci(
            vk::PipelineVertexInputStateCreateFlags(),
            static_cast<uint32_t>(vtx_inp_binding_descs.size()),
            DataSafety(vtx_inp_binding_descs),
            static_cast<uint32_t>(vtx_inp_attrib_descs.size()),
            DataSafety(vtx_inp_attrib_descs));

    // Input assembly state create info
    vk::PipelineInputAssemblyStateCreateInfo inp_assembly_state_ci(
            vk::PipelineInputAssemblyStateCreateFlags(),
            pipeline_info.prim_type);

    // Viewport state create info (Not static)
    vk::PipelineViewportStateCreateInfo viewport_state_ci(
            vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);

    // Rasterization state create info
    vk::PipelineRasterizationStateCreateInfo rasterization_state_ci(
            vk::PipelineRasterizationStateCreateFlags(),
            false,                   // depthClampEnable
            false,                   // rasterizerDiscardEnable
            vk::PolygonMode::eFill,  // polygonMode
            pipeline_info.face_culling,
            vk::FrontFace::eClockwise,  // frontFace
            false,                      // depthBiasEnable
            0.0f,                       // depthBiasConstantFactor
            0.0f,                       // depthBiasClamp
            0.0f,                       // depthBiasSlopeFactor
            pipeline_info.line_width);

    // Multi-sample state create info
    vk::PipelineMultisampleStateCreateInfo multisample_state_ci;

    // Depth stencil state create info
    vk::StencilOpState stencil_op(vk::StencilOp::eKeep, vk::StencilOp::eKeep,
                                  vk::StencilOp::eKeep, vk::CompareOp::eAlways);
    vk::PipelineDepthStencilStateCreateInfo depth_stencil_state_ci(
            vk::PipelineDepthStencilStateCreateFlags(),
            pipeline_info.depth_test_enable, pipeline_info.depth_write_enable,
            pipeline_info.depth_comp_op,
            false,       // depthBoundTestEnable
            false,       // stencilTestEnable
            stencil_op,  // front
            stencil_op   // back
    );

    // Color blend attachment state
    std::vector<vk::PipelineColorBlendAttachmentState> blend_attach_states;
    for (auto &&color_blend_info : pipeline_info.color_blend_infos) {
        blend_attach_states.emplace_back(
                color_blend_info.blend_enable,
                color_blend_info.blend_src_col_factor,
                color_blend_info.blend_dst_col_factor,
                color_blend_info.blend_color_op,
                color_blend_info.blend_src_alpha_factor,
                color_blend_info.blend_dst_alpha_factor,
                color_blend_info.blend_alpha_op,
                color_blend_info.blend_write_mask);
    }
    // Color blend attachment state create info
    vk::PipelineColorBlendStateCreateInfo color_blend_state_ci(
            vk::PipelineColorBlendStateCreateFlags(),  // flags
            false,                                     // logicOpEnable
            vk::LogicOp::eNoOp,                        // logicOp
            static_cast<uint32_t>(blend_attach_states.size()),
            blend_attach_states.data(),
            {{1.0f, 1.0f, 1.0f, 1.0f}}  // blendConstants
    );

    // Dynamic states
    const vk::DynamicState dynamic_states[2] = {vk::DynamicState::eViewport,
                                                vk::DynamicState::eScissor};
    // Dynamic state create info
    vk::PipelineDynamicStateCreateInfo dynamic_state_ci(
            vk::PipelineDynamicStateCreateFlags(), 2, dynamic_states);

    // Repack descriptor set layout
    std::vector<vk::DescriptorSetLayout> desc_set_layouts;
    desc_set_layouts.reserve(desc_set_packs.size());
    for (auto &&desc_set_pack : desc_set_packs) {
        desc_set_layouts.push_back(desc_set_pack->desc_set_layout.get());
    }
    // Create pipeline layout
    auto pipeline_layout = device->createPipelineLayoutUnique(
            {vk::PipelineLayoutCreateFlags(),
             static_cast<uint32_t>(desc_set_layouts.size()),
             desc_set_layouts.data()});

    // Create pipeline
    auto pipeline = device->createGraphicsPipelineUnique(
            nullptr,
            {vk::PipelineCreateFlags(),
             static_cast<uint32_t>(shader_stage_cis.size()),
             shader_stage_cis.data(), &vtx_inp_state_ci, &inp_assembly_state_ci,
             nullptr, &viewport_state_ci, &rasterization_state_ci,
             &multisample_state_ci, &depth_stencil_state_ci,
             &color_blend_state_ci, &dynamic_state_ci, pipeline_layout.get(),
             render_pass_pack->render_pass.get(), subpass_idx});

    return PipelinePackPtr(
            new PipelinePack{std::move(pipeline_layout), std::move(pipeline)});
}

// -----------------------------------------------------------------------------
// ------------------------------- Command Buffer ------------------------------
// -----------------------------------------------------------------------------
CommandBuffersPackPtr CreateCommandBuffersPack(const vk::UniqueDevice &device,
                                               uint32_t queue_family_idx,
                                               uint32_t n_cmd_buffers,
                                               bool reset_enable) {
    // Create flags
    vk::CommandPoolCreateFlags flags;
    if (reset_enable) {
        flags |= vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    }

    // Create a command pool
    vk::UniqueCommandPool cmd_pool =
            device->createCommandPoolUnique({flags, queue_family_idx});

    // Allocate a command buffer from the command pool
    auto cmd_bufs = device->allocateCommandBuffersUnique(
            {cmd_pool.get(), vk::CommandBufferLevel::ePrimary, n_cmd_buffers});

    return CommandBuffersPackPtr(
            new CommandBuffersPack{std::move(cmd_pool), std::move(cmd_bufs)});
}

void BeginCommand(const vk::UniqueCommandBuffer &cmd_buf,
                  bool one_time_submit) {
    // Create begin flags
    vk::CommandBufferUsageFlags flags;
    if (one_time_submit) {
        flags |= vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    }
    // Begin
    cmd_buf->begin({flags});
}

void EndCommand(const vk::UniqueCommandBuffer &cmd_buf) {
    // End
    cmd_buf->end();
}

void ResetCommand(const vk::UniqueCommandBuffer &cmd_buf) {
    // Reset
    cmd_buf->reset(vk::CommandBufferResetFlags());
}

void CmdBeginRenderPass(const vk::UniqueCommandBuffer &cmd_buf,
                        const RenderPassPackPtr &render_pass_pack,
                        const FrameBufferPackPtr &frame_buffer_pack,
                        const std::vector<vk::ClearValue> &clear_vals,
                        const vk::Rect2D &render_area_org) {
    // Decide render area
    vk::Rect2D render_area;
    if (0 < render_area_org.extent.width && 0 < render_area_org.extent.height) {
        render_area = render_area_org;
    } else {
        // Default size is same with surface size
        render_area.extent.width = frame_buffer_pack->width;
        render_area.extent.height = frame_buffer_pack->height;
    }

    // Begin render pass
    cmd_buf->beginRenderPass(
            {render_pass_pack->render_pass.get(),
             frame_buffer_pack->frame_buffer.get(), render_area,
             static_cast<uint32_t>(clear_vals.size()), DataSafety(clear_vals)},
            vk::SubpassContents::eInline);
}

void CmdNextSubPass(const vk::UniqueCommandBuffer &cmd_buf) {
    cmd_buf->nextSubpass(vk::SubpassContents::eInline);
}

void CmdEndRenderPass(const vk::UniqueCommandBuffer &cmd_buf) {
    cmd_buf->endRenderPass();
}

void CmdBindPipeline(const vk::UniqueCommandBuffer &cmd_buf,
                     const PipelinePackPtr &pipeline_pack,
                     const vk::PipelineBindPoint &bind_point) {
    cmd_buf->bindPipeline(bind_point, pipeline_pack->pipeline.get());
}

void CmdBindDescSets(const vk::UniqueCommandBuffer &cmd_buf,
                     const PipelinePackPtr &pipeline_pack,
                     const std::vector<DescSetPackPtr> &desc_set_packs,
                     const std::vector<uint32_t> &dynamic_offsets,
                     const vk::PipelineBindPoint &bind_point) {
    // Repack descriptor set layout
    std::vector<vk::DescriptorSet> desc_sets;
    desc_sets.reserve(desc_set_packs.size());
    for (auto &&desc_set_pack : desc_set_packs) {
        desc_sets.push_back(desc_set_pack->desc_set.get());
    }
    // Bind
    const uint32_t first_set = 0;
    cmd_buf->bindDescriptorSets(
            bind_point, pipeline_pack->pipeline_layout.get(), first_set,
            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(),
            static_cast<uint32_t>(dynamic_offsets.size()),
            DataSafety(dynamic_offsets));
}

void CmdBindVertexBuffers(const vk::UniqueCommandBuffer &cmd_buf,
                          const std::vector<BufferPackPtr> &vtx_buf_packs) {
    // Repack buffers
    std::vector<vk::Buffer> vtx_bufs;
    vtx_bufs.reserve(vtx_buf_packs.size());
    for (auto &&vtx_buf_pack : vtx_buf_packs) {
        vtx_bufs.push_back(vtx_buf_pack->buf.get());
    }
    // Bind
    const uint32_t n_vtx_bufs = static_cast<uint32_t>(vtx_bufs.size());
    const uint32_t first_binding = 0;
    const std::vector<vk::DeviceSize> offsets(n_vtx_bufs, 0);  // no offsets
    cmd_buf->bindVertexBuffers(first_binding, n_vtx_bufs, vtx_bufs.data(),
                               offsets.data());
}

void CmdSetViewport(const vk::UniqueCommandBuffer &cmd_buf,
                    const vk::Viewport &viewport) {
    // Set a viewport
    const uint32_t first_viewport = 0;
    cmd_buf->setViewport(first_viewport, 1, &viewport);
}

void CmdSetViewport(const vk::UniqueCommandBuffer &cmd_buf,
                    const vk::Extent2D &viewport_size) {
    // Convert to viewport
    const float min_depth = 0.f;
    const float max_depth = 1.f;
    vk::Viewport viewport(0.f, 0.f, static_cast<float>(viewport_size.width),
                          static_cast<float>(viewport_size.height), min_depth,
                          max_depth);
    // Set
    CmdSetViewport(cmd_buf, viewport);
}

void CmdSetScissor(const vk::UniqueCommandBuffer &cmd_buf,
                   const vk::Rect2D &scissor) {
    // Set a scissor
    const uint32_t first_scissor = 0;
    cmd_buf->setScissor(first_scissor, 1, &scissor);
}

void CmdSetScissor(const vk::UniqueCommandBuffer &cmd_buf,
                   const vk::Extent2D &scissor_size) {
    // Convert to rect2D
    vk::Rect2D rect(vk::Offset2D(0, 0), scissor_size);
    // Set
    CmdSetScissor(cmd_buf, rect);
}

void CmdDraw(const vk::UniqueCommandBuffer &cmd_buf, uint32_t n_vtxs,
             uint32_t n_instances) {
    const uint32_t first_vtx = 0;
    const uint32_t first_instance = 0;
    cmd_buf->draw(n_vtxs, n_instances, first_vtx, first_instance);
}

// -----------------------------------------------------------------------------
// ----------------------------------- Queue -----------------------------------
// -----------------------------------------------------------------------------
vk::Queue GetQueue(const vk::UniqueDevice &device, uint32_t queue_family_idx,
                   uint32_t queue_idx) {
    return device->getQueue(queue_family_idx, queue_idx);
}

void QueueSubmit(const vk::Queue &queue, const vk::UniqueCommandBuffer &cmd_buf,
                 const FencePtr &signal_fence,
                 const std::vector<WaitSemaphoreInfo> &wait_semaphore_infos,
                 const std::vector<SemaphorePtr> &signal_semaphores) {
    const uint32_t n_cmd_bufs = 1;
    const uint32_t n_wait_semaphores =
            static_cast<uint32_t>(wait_semaphore_infos.size());
    const uint32_t n_signal_semaphores =
            static_cast<uint32_t>(signal_semaphores.size());

    // Unpack wait semaphore infos
    std::vector<vk::Semaphore> wait_semaphores_raw;
    std::vector<vk::PipelineStageFlags> wait_semaphore_stage_flags_raw;
    wait_semaphores_raw.reserve(n_wait_semaphores);
    wait_semaphore_stage_flags_raw.reserve(n_wait_semaphores);
    for (auto &&info : wait_semaphore_infos) {
        wait_semaphores_raw.push_back(std::get<0>(info)->get());
        wait_semaphore_stage_flags_raw.push_back(std::get<1>(info));
    }

    // Unpack signal semaphores
    std::vector<vk::Semaphore> signal_semaphores_raw;
    signal_semaphores_raw.reserve(n_signal_semaphores);
    for (auto &&s : signal_semaphores) {
        signal_semaphores_raw.push_back(s->get());
    }

    // Resolve signal fence
    vk::Fence fence = signal_fence ? signal_fence->get() : vk::Fence();

    // Submit
    vk::SubmitInfo submit_info = {n_wait_semaphores,
                                  DataSafety(wait_semaphores_raw),
                                  DataSafety(wait_semaphore_stage_flags_raw),
                                  n_cmd_bufs,
                                  &cmd_buf.get(),
                                  n_signal_semaphores,
                                  DataSafety(signal_semaphores_raw)};
    queue.submit(1, &submit_info, fence);
}

void QueuePresent(const vk::Queue &queue,
                  const SwapchainPackPtr &swapchain_pack, uint32_t img_idx,
                  const std::vector<SemaphorePtr> &wait_semaphores) {
    // Unpack signal semaphores
    const uint32_t n_wait_semaphores =
            static_cast<uint32_t>(wait_semaphores.size());
    std::vector<vk::Semaphore> wait_semaphores_raw;
    wait_semaphores_raw.reserve(wait_semaphores.size());
    for (auto &&s : wait_semaphores) {
        wait_semaphores_raw.push_back(s->get());
    }

    // Present
    const uint32_t n_swapchains = 1;
    queue.presentKHR({n_wait_semaphores, DataSafety(wait_semaphores_raw),
                      n_swapchains, &swapchain_pack->swapchain.get(),
                      &img_idx});
}

}  // namespace vkw
