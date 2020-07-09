#include <bits/stdint-uintn.h>
#include "../example/01_rotate_box/app.h"
#include "../example/02_load_obj/app.h"
#include "../example/03_load_obj_many/app.h"
#include "../example/04_deferred_shading/app.h"
#include "../example/05_instancing/app.h"
#include "../example/06_raster_vtx_id/app.h"
#include "../example/07_image_transfer/app.h"
#include "../example/08_comp_shader/app.h"
#include "../example/09_inverse_uv/app.h"

#include <sstream>
#include <functional>
#include <vector>

// List of application functions
const std::vector<std::function<void(const vkw::WindowPtr& window, std::function<void()>)>> APP_FUNCS = {RunExampleApp01, RunExampleApp02, RunExampleApp03,
                        RunExampleApp04, RunExampleApp05, RunExampleApp06,
                        RunExampleApp07, RunExampleApp08, RunExampleApp09,
};

template <typename T>
std::string AsStr(const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

uint32_t DecideAppID(int argc, char const* argv[]) {
    // Decide application ID
    uint32_t app_id = 0;
    if (2 <= argc) {
        std::istringstream iss;
        app_id = static_cast<uint32_t>(std::atoi(argv[1]));
    }
    return app_id;
}

int main(int argc, char const* argv[]) {
    // Decide application ID
    const uint32_t app_id = DecideAppID(argc, argv);
    vkw::PrintInfo("Application Index: " + AsStr(app_id));

    // Common variables
    auto gen_window = [&]() {
        return vkw::InitGLFWWindow("VKW Example " + AsStr(app_id), 600, 600);
    };
    auto window = gen_window();
    auto draw_hook = [&]() {
        if (glfwWindowShouldClose(window.get())) {
            window = nullptr;
            throw std::runtime_error("GLFW should be closed");
        }
        glfwPollEvents();
    };

    // Run application
    try {
        if (app_id == 0) {
            // Run all applications
            for (uint32_t i = 0; i < APP_FUNCS.size(); i++) {
                vkw::PrintInfo("Run application: " + AsStr(app_id));
                try {
                    if (!window) {
                        window = gen_window();
                    }
                    APP_FUNCS[i](window, draw_hook);
                } catch(const std::exception& e) {
                    vkw::PrintInfo(e.what());
                }
            }
        } else if (app_id <= APP_FUNCS.size()) {
            // Run one application
            APP_FUNCS[app_id + 1](window, draw_hook);
        } else {
            vkw::PrintErr("Invalid application ID");
        }
    } catch(const std::exception& e) {
        vkw::PrintInfo(e.what());
        vkw::PrintInfo("Exit app");
    }

    return 0;
}
