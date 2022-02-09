#include "../example/01_rotate_box/app.h"
#include "../example/02_load_obj/app.h"
#include "../example/03_load_obj_many/app.h"
#include "../example/04_deferred_shading/app.h"
#include "../example/05_instancing/app.h"
#include "../example/06_raster_vtx_id/app.h"

#include <jni.h>
#include <thread>

vkw::WindowPtr window;

extern "C" JNIEXPORT void JNICALL
Java_com_takiyu_vulkanwrapperexample_MainActivity_nativeSetSurface(
        JNIEnv* jenv, jclass jclazz, jobject jsurface) {

    if (jsurface == 0) {
        window.reset();
        return;
    }

    // Common variables
    window = vkw::InitANativeWindow(jenv, jsurface);
    auto draw_hook = [&]() {
        if (!window) {
            throw std::runtime_error("ANativeWindow should be closed");
        }
    };

    std::thread thread([&]() {
        // Run application
        try {
            RunExampleApp01(window, draw_hook);
            // RunExampleApp02(window, draw_hook);
            // RunExampleApp03(window, draw_hook);
            // RunExampleApp04(window, draw_hook);
            // RunExampleApp05(window, draw_hook);
            // RunExampleApp06(window, draw_hook);
        } catch(const std::exception& e) {
            vkw::PrintInfo(e.what());
            vkw::PrintInfo("Exit app");
        }
    });
    thread.detach();
}
