#include <jni.h>
#include <android/native_window.h> // requires ndk r5 or newer
#include <android/native_window_jni.h> // requires ndk r5 or newer

#include <vkw/vkw.h>

#include <string>

static ANativeWindow *window = nullptr;
vk::UniqueInstance g_instance;
vk::UniqueSurfaceKHR g_surface;

extern "C"
JNIEXPORT void JNICALL
Java_com_imailab_vulkanwrapperexample_MainActivity_nativeSetSurface(JNIEnv *jenv, jclass clazz,
                                                                    jobject surface) {
    if (surface != 0) {
        window = ANativeWindow_fromSurface(jenv, surface);

        const std::string app_name = "app name";
        const int app_version = 1;
        const std::string engine_name = "engine name";
        const int engine_version = 1;
        uint32_t win_w = 600;
        uint32_t win_h = 600;

        g_instance = vkw::CreateInstance(app_name, app_version, engine_name,
                                         engine_version);

        if (window) {
            g_surface = vkw::CreateSurface(g_instance, window);
        }
    } else {
        g_surface.reset();
        g_instance.reset();
        ANativeWindow_release(window);
        window = nullptr;
        return;
    }

}
