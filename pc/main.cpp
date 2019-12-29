#include "../example/01_rotate_box/app.h"


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
        RunExampleApp01(window, draw_hook);
    } catch(...) {
        vkw::PrintInfo("Exit app");
    }

    return 0;
}
