#include "../example/01_rotate_box/app.h"
#include "../example/02_load_obj/app.h"
#include "../example/03_load_obj_many/app.h"
#include "../example/04_deferred_shading/app.h"
#include "../example/05_instancing/app.h"
#include "../example/06_raster_vtx_id/app.h"
#include "../example/07_compute_shader/app.h"

#include <sstream>


int main(int argc, char const* argv[]) {
    // Decide application ID
    int app_id = 1;
    if (2 <= argc) {
        std::istringstream iss;
        app_id = std::atoi(argv[1]);
    }
    // Print application ID
    std::stringstream app_id_ss;
    app_id_ss << app_id;
    const std::string& app_id_str = app_id_ss.str();
    vkw::PrintInfo("Application Index: " + app_id_str);

    // Common variables
    auto window = vkw::InitGLFWWindow("VKW Example " + app_id_str, 600, 600);
    auto draw_hook = [&]() {
        if (glfwWindowShouldClose(window.get())) {
            throw std::runtime_error("GLFW should be closed");
        }
        glfwPollEvents();
    };

    // Run application
    try {
        if (app_id == 1) {
            RunExampleApp01(window, draw_hook);
        } else if (app_id == 2) {
            RunExampleApp02(window, draw_hook);
        } else if (app_id == 3) {
            RunExampleApp03(window, draw_hook);
        } else if (app_id == 4) {
            RunExampleApp04(window, draw_hook);
        } else if (app_id == 5) {
            RunExampleApp05(window, draw_hook);
        } else if (app_id == 6) {
            RunExampleApp06(window, draw_hook);
        } else if (app_id == 7) {
            RunExampleApp07(window, draw_hook);
        } else {
            vkw::PrintErr("Invalid application ID");
        }
    } catch(const std::exception& e) {
        vkw::PrintInfo(e.what());
        vkw::PrintInfo("Exit app");
    }

    return 0;
}
