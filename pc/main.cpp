#include <functional>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iostream>

#include "../example/01_rotate_box/app.h"
#include "../example/02_load_obj/app.h"
#include "../example/03_load_obj_many/app.h"
#include "../example/04_deferred_shading/app.h"
#include "../example/05_instancing/app.h"
#include "../example/06_raster_vtx_id/app.h"
#include "../example/07_image_transfer/app.h"
#include "../example/08_comp_shader/app.h"
#include "../example/09_inverse_uv/app.h"
#include "../example/10_glsl_optim/app.h"
#include "../example/11_img_buf/app.h"
#include "../example/12_comp_shader_atomic_float/app.h"
#include "vkw/vkw.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
namespace {

// List of application functions
using APP_FUNC_TYPE = std::function<void(const vkw::WindowPtr& window,
                                         std::function<void()>)>;
const std::vector<APP_FUNC_TYPE> APP_FUNCS = {
        RunExampleApp01, RunExampleApp02, RunExampleApp03,
        RunExampleApp04, RunExampleApp05, RunExampleApp06,
        RunExampleApp07, RunExampleApp08, RunExampleApp09,
        RunExampleApp10, RunExampleApp11, RunExampleApp12,
};

// Window
static vkw::WindowPtr g_window;

void InitWindow(const std::string& title) {
    g_window = vkw::InitGLFWWindow(title, 600, 600);
}

void DrawHook() {
    if (glfwWindowShouldClose(g_window.get())) {
        g_window = nullptr;
        throw std::runtime_error("GLFW should be closed");
    }
    glfwPollEvents();
}

#include <iostream>
static uint32_t g_draw_cnt = 0;
const uint32_t N_LIMITED_DRAW_MAX = 30;
void LimitedDrawHook() {
    // Draw normally
    DrawHook();
    // Check drawing count
    g_draw_cnt++;
    if (N_LIMITED_DRAW_MAX < g_draw_cnt) {
        g_draw_cnt = 0;
        throw std::runtime_error("Go to next application");
    }
}

// String utility
template <typename T>
std::string AsStr(const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

// Argument
uint32_t DecideAppID(int argc, char const* argv[]) {
    // Decide application ID
    uint32_t app_id = 0;
    if (2 <= argc) {
        std::istringstream iss;
        app_id = static_cast<uint32_t>(std::atoi(argv[1]));
    }
    return app_id;
}

}  // namespace
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int main(int argc, char const* argv[]) {
    // Decide application ID
    const uint32_t app_id = DecideAppID(argc, argv);
    vkw::PrintInfo("Application Index: " + AsStr(app_id));

    // Run application
    try {
        if (app_id == 0) {
            // Run all applications
            for (uint32_t i = 1; i < APP_FUNCS.size() + 1; i++) {
                vkw::PrintInfo("Run application: " + AsStr(i));
                try {
                    // Create window
                    InitWindow("Example App " + AsStr(i));
                    // Run
                    APP_FUNCS[i - 1](g_window, LimitedDrawHook);
                } catch (const std::exception& e) {
                    vkw::PrintInfo(e.what());
                    // Go to next application
                }
            }
            vkw::PrintErr("All applications have finished");
        } else if (app_id <= APP_FUNCS.size()) {
            // Run one application
            InitWindow("Example App " + AsStr(app_id));
            APP_FUNCS[app_id - 1](g_window, DrawHook);
        } else {
            vkw::PrintErr("Invalid application ID");
        }
    } catch (const std::exception& e) {
        vkw::PrintInfo(e.what());
        vkw::PrintInfo("Exit app");
    }

    return 0;
}
