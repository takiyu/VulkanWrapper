#include <vkw/vkw_deferer.h>

#include <unordered_set>
#include <unordered_map>
#include <iostream>

namespace vkw {

namespace {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
template <typename T>
std::vector<T> CastToVec(const std::unordered_map<uint32_t, T> idxed_map) {
    // Find largest index
    uint32_t max_idx = 0;
    for (auto&& it : idxed_map) {
        max_idx = std::max(max_idx, it.first);
    }

    // Cast
    std::vector<T> result(max_idx + 1);
    for (auto&& it : idxed_map) {
        result[it.first] = it.second;
    }

    return result;
}

auto CreateDescriptorSetPair(const ContextPtr& vk_ctx, const DefererStageInfo& ci) {
    // Create descriptor set
    std::unordered_map<uint32_t, vkw::DescSetInfo> desc_set_infos;
    // Set input attachment info
    if (0 < ci.inp_imgs.size()) {
        desc_set_infos[ci.inp_imgs_binding_idx] =
            {vk::DescriptorType::eInputAttachment, ci.inp_imgs.size(),
                                    vk::ShaderStageFlagBits::eFragment};
    }
    // Set texture info
    if (0 < ci.inp_texs.size()) {
        desc_set_infos[ci.inp_texs_binding_idx] =
        {vk::DescriptorType::eCombinedImageSampler,
                ci.inp_texs.size(), vk::ShaderStageFlagBits::eAllGraphics};
    }
    // Set dynamic uniform info
    if (0 < ci.dyn_unif_bufs.size()) {
        desc_set_infos[ci.dyn_unif_bufs_binding_idx] =
        {vk::DescriptorType::eUniformBufferDynamic,
                ci.dyn_unif_bufs.size(),
                vk::ShaderStageFlagBits::eAllGraphics};
    }
    // Create a descriptor set
    auto desc_set_pack = vkw::CreateDescriptorSetPack(vk_ctx->m_device,
                                                      CastToVec(desc_set_infos));

    // Add write descriptor set
    auto write_desc_set_pack = vkw::CreateWriteDescSetPack();
    // Set input attachment images
    if (0 < ci.inp_imgs.size()) {
        vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack,
                             ci.inp_imgs_binding_idx, ci.inp_imgs,
                             vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    // Set texture inputs
    if (0 < ci.inp_texs.size()) {
        vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack,
                             ci.inp_texs_binding_idx, ci.inp_texs,
                             vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    // Set dynamic uniform info
    if (0 < ci.dyn_unif_bufs.size()) {
        vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack,
                             ci.dyn_unif_bufs_binding_idx,
                             ci.dyn_unif_bufs);
    }
    // Write
    vkw::UpdateDescriptorSets(vk_ctx->m_device, write_desc_set_pack);

    return std::make_tuple(desc_set_pack, write_desc_set_pack);
}
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
}

// -----------------------------------------------------------------------------
// ---------------------------------- Deferer ----------------------------------
// -----------------------------------------------------------------------------
void Deferer::init(const ContextPtr& vk_ctx, const vk::Extent2D& size,
                   const std::vector<DefererStageInfo>& staged_cis) {
    // Set basic infos
    m_vk_ctx = vk_ctx;
    m_n_stages = static_cast<uint32_t>(staged_cis.size());
    m_size = size;

    // Create descriptor sets
    m_desc_set_packs.clear();
    m_desc_set_packs.reserve(m_n_stages);
    m_write_desc_set_packs.clear();
    m_write_desc_set_packs.reserve(m_n_stages);
    for (auto&& ci : staged_cis) {
        // Create
        auto desc_set_pair = CreateDescriptorSetPair(m_vk_ctx, ci);
        // Register
        m_desc_set_packs.push_back(std::move(std::get<0>(desc_set_pair)));
        m_write_desc_set_packs.push_back(std::move(std::get<1>(desc_set_pair)));
    }

    // Resolve dependency
    std::unordered_map<vkw::ImagePackPtr, uint32_t> out_img_stages;
    for (uint32_t stage_idx = 0; stage_idx < m_n_stages; stage_idx++) {
        const auto& out_imgs = staged_cis[stage_idx].out_imgs;
        for (auto&& out_img : out_imgs) {
            out_img_stages[out_img] = stage_idx;
        }
    }
    std::cout << out_img_stages.size() << std::endl;

    // Create render pass
    m_render_pass_packs.clear();
    m_render_pass_packs.reserve(m_n_stages);
    for (auto&& ci : staged_cis) {
//     // Create render pass
//     auto render_pass_pack = vkw::CreateRenderPassPack();
//     // 0) Add color attachment for surface
//     vkw::AddAttachientDesc(
//             render_pass_pack, surface_format, vk::AttachmentLoadOp::eClear,
//             vk::AttachmentStoreOp::eStore, vk::ImageLayout::ePresentSrcKHR);
//     // 1) Add gbuffer 0 (color) attachment
//     vkw::AddAttachientDesc(render_pass_pack, gbuf_col_format,
//                            vk::AttachmentLoadOp::eClear,
//                            vk::AttachmentStoreOp::eStore,
//                            vk::ImageLayout::eColorAttachmentOptimal);
//     // 2) Add gbuffer 1 (normal) attachment
//     vkw::AddAttachientDesc(render_pass_pack, gbuf_nor_format,
//                            vk::AttachmentLoadOp::eClear,
//                            vk::AttachmentStoreOp::eStore,
//                            vk::ImageLayout::eColorAttachmentOptimal);
//     // 3) Add depth attachment
//     vkw::AddAttachientDesc(render_pass_pack, depth_format,
//                            vk::AttachmentLoadOp::eClear,
//                            vk::AttachmentStoreOp::eDontCare,
//                            vk::ImageLayout::eDepthStencilAttachmentOptimal);
//     // Add subpass 1
//     vkw::AddSubpassDesc(render_pass_pack,
//                         {
//                                 // No input attachments
//                         },
//                         {
//                                 {1, vk::ImageLayout::eColorAttachmentOptimal},
//                                 {2, vk::ImageLayout::eColorAttachmentOptimal},
//                         },
//                         {3, vk::ImageLayout::eDepthStencilAttachmentOptimal});
//     // Add subpass 2
//     vkw::AddSubpassDesc(render_pass_pack,
//                         {
//                                 {1, vk::ImageLayout::eShaderReadOnlyOptimal},
//                                 {2, vk::ImageLayout::eShaderReadOnlyOptimal},
//                         },
//                         {
//                                 {0, vk::ImageLayout::eColorAttachmentOptimal},
//                         });  // No depth
//     // Add dependency
//     vkw::AddSubpassDepend(render_pass_pack,
//                           {0, vk::PipelineStageFlagBits::eColorAttachmentOutput,
//                            vk::AccessFlagBits::eColorAttachmentWrite},
//                           {1, vk::PipelineStageFlagBits::eFragmentShader,
//                            vk::AccessFlagBits::eInputAttachmentRead},
//                           vk::DependencyFlagBits::eByRegion);
//     // Create render pass instance
//     vkw::UpdateRenderPass(device, render_pass_pack);
// 
//     // Create frame buffers for swapchain images
//     auto frame_buffer_packs = vkw::CreateFrameBuffers(
//             device, render_pass_pack,
//             {nullptr, gbuf_col_img_pack, gbuf_nor_img_pack, depth_img_pack},
//             swapchain_pack);
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

}  // namespace vkw
