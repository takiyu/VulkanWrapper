#include "../example/01_rotate_box/app.h"
#include "../example/02_load_obj/app.h"
#include "../example/03_load_obj_many/app.h"
#include "../example/04_deferred_shading/app.h"
#include "../example/05_raster_vtx_id/app.h"


int main(int argc, char const* argv[]) {
    (void)argc, (void)argv;

    // Common variables
    auto window = vkw::InitGLFWWindow("VKW Example", 600, 600);
    auto draw_hook = [&]() {
        if (glfwWindowShouldClose(window.get())) {
            throw std::runtime_error("GLFW should be closed");
        }
        glfwPollEvents();
    };

    // Run application
    try {
        // RunExampleApp01(window, draw_hook);
        // RunExampleApp02(window, draw_hook);
        // RunExampleApp03(window, draw_hook);
        // RunExampleApp04(window, draw_hook);
        RunExampleApp05(window, draw_hook);
    } catch(const std::exception& e) {
        vkw::PrintInfo(e.what());
        vkw::PrintInfo("Exit app");
    }

    return 0;
}
