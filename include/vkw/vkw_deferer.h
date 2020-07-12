#ifndef VKW_DEFERER_H_20200625
#define VKW_DEFERER_H_20200625

#include <vkw/vkw.h>
#include <vkw/vkw_context.h>

#include <unordered_map>

namespace vkw {

// -----------------------------------------------------------------------------
// ---------------------------- Deferer Stage Info -----------------------------
// -----------------------------------------------------------------------------
struct DefererStageInfo {
    // Input/output
    uint32_t inp_imgs_binding_idx = 0;
    std::vector<vkw::ImagePackPtr> inp_imgs;
    uint32_t inp_texs_binding_idx = 1;
    std::vector<vkw::TexturePackPtr> inp_texs;
    uint32_t dyn_unif_bufs_binding_idx = 2;
    std::vector<vkw::BufferPackPtr> dyn_unif_bufs;
    std::vector<vkw::ImagePackPtr> out_imgs;
    vkw::ImagePackPtr depth_img;
    // Vertex input
    std::vector<VtxInputBindingInfo> vtx_inp_binding_infos;
    std::vector<VtxInputAttribInfo> vtx_inp_attrib_infos;
    // Pipeline
    PipelineInfo pipeline_info;
    // Shader
    std::string vert_shader_code;
    std::string frag_shader_code;
};


// -----------------------------------------------------------------------------
// ---------------------------------- Deferer ----------------------------------
// -----------------------------------------------------------------------------
class Deferer;
using DefererPtr = std::shared_ptr<Deferer>;
class Deferer {
public:
    // -------------------------------------------------------------------------
    // ---------------------------- Creator Method -----------------------------
    // -------------------------------------------------------------------------
    template <typename... T>
    static auto Create(T&&... args) {
        return DefererPtr(new Deferer(std::forward<T>(args)...));
    }

    // -------------------------------------------------------------------------
    // -------------------------------- Methods --------------------------------
    // -------------------------------------------------------------------------
    void init(const ContextPtr& vk_ctx, const vk::Extent2D& size,
              const std::vector<DefererStageInfo>& staged_cis);

    // -------------------------------------------------------------------------
    // ------------------------ Public Member Variables ------------------------
    // -------------------------------------------------------------------------
    ContextPtr m_vk_ctx;
    uint32_t m_n_stages;
    vk::Extent2D m_size;
    std::vector<DescSetPackPtr> m_desc_set_packs;
    std::vector<WriteDescSetPackPtr> m_write_desc_set_packs;
    std::vector<RenderPassPack> m_render_pass_packs;

    // -------------------------------------------------------------------------
    // ------------------------- Default Shader Codes --------------------------
    // -------------------------------------------------------------------------
    static constexpr const char* DEFAULT_VERT_SRC = R"(
        #version 460
        void main() {
            vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
            vec2 screen_pos = uv * 2.0f - 1.0f;
            gl_Position = vec4(screen_pos, 0.0f, 1.0f);
        }
    )";

    static constexpr const char* DEFAULT_FRAG_SRC = R"(
        #version 460
        layout (location = 0) out vec4 frag_color;
        void main() {
            frag_color = vec4(gl_FragCoord.xy, 0.0, 1.0);
        }
    )";

    // -------------------------------------------------------------------------
    // ----------------------------- Constructors ------------------------------
    // -------------------------------------------------------------------------
private:
    Deferer() {}

    template <typename... T>
    Deferer(T&&... args) {
        init(std::forward<T>(args)...);
    }

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

}  // namespace vkw

#endif  // end of include guard
