message(STATUS "--------------------------------------------------------------")
message(STATUS "-------------------------- Begin VKW -------------------------")
message(STATUS "--------------------------------------------------------------")
cmake_minimum_required(VERSION 3.10)

# ------------------------------------------------------------------------------
# -------------------------------- Check parent --------------------------------
# ------------------------------------------------------------------------------
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if (HAS_PARENT)
    message(STATUS "Build VKW as subdirectory mode")
else()
    message(STATUS "Build VKW as independent mode")
endif()

# ------------------------------------------------------------------------------
# ----------------------------------- Options ----------------------------------
# ------------------------------------------------------------------------------
set(VKW_BUILD_EXAMPLES ON CACHE BOOL "Build examples")
set(VKW_LINK_DYNAMIC_VK ON CACHE BOOL "Dynamic link for vulkan.")
set(VKW_USE_VULKAN_LOADER ON CACHE BOOL "Use Vulkan-Loader when dynamic link.")
set(VKW_USE_INTERNAL_GLFW ON CACHE BOOL "Use Internal GLFW.")
set(VKW_EXTERNAL_INCLUDE "" CACHE STRING "External include paths.")
set(VKW_EXTERNAL_LIBRARY "" CACHE STRING "External libraries.")
set(VKW_THIRD_PARTY_PATH "" CACHE STRING "Third party directory path.")

# Handle flag dependency
if (NOT VKW_LINK_DYNAMIC_VK OR MSVC OR MINGW OR ANDROID)
    set(VKW_USE_VULKAN_LOADER OFF)  # No vulkan-loader for static Vulkan
endif()
if (ANDROID)
    set(VKW_USE_INTERNAL_GLFW OFF)  # No GLFW for Android
endif()
if (NOT VKW_THIRD_PARTY_PATH)
    set(VKW_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

# External includes/libraries
list(APPEND VKW_INCLUDE ${VKW_EXTERNAL_INCLUDE})
list(APPEND VKW_LIBRARY ${VKW_EXTERNAL_LIBRARY})

# ------------------------------------------------------------------------------
# ----------------------------------- Common -----------------------------------
# ------------------------------------------------------------------------------
project(vkw CXX C)
set(CMAKE_CXX_STANDARD 14)  # C++ 14

# Include common configurations
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common_setups.cmake)

# ------------------------------------------------------------------------------
# --------------------------- Third Party Libraries ----------------------------
# ------------------------------------------------------------------------------
list(APPEND VKW_INCLUDE ${VKW_THIRD_PARTY_PATH})

if (VKW_LINK_DYNAMIC_VK)
    # Vulkan (Dynamic)
    message(STATUS "Build VKW as dynacmic-vk mode")
    setup_third_party("https://github.com/KhronosGroup/Vulkan-Headers"
                      "7fe877c90abf00bc71b3c68f49db4c9bb1010411" FALSE
                      ${VKW_THIRD_PARTY_PATH})
    list(APPEND VKW_INCLUDE ${VKW_THIRD_PARTY_PATH}/Vulkan-Headers/include)
    list(APPEND VKW_DEFINE -DVULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1  # dlopen
                           -DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)  # dispatch

    if (VKW_USE_VULKAN_LOADER)
        # Vulkan-Loader
        setup_third_party("https://github.com/KhronosGroup/Vulkan-Loader"
                          "b83102e3d5b8929c09989e0b879fe6ac5077e013" TRUE
                          ${VKW_THIRD_PARTY_PATH})
        set(VulkanHeaders_INCLUDE_DIR
                ${VKW_THIRD_PARTY_PATH}/Vulkan-Headers/include
                CACHE INTERNAL "" FORCE)
        set(VulkanRegistry_DIR
                ${VKW_THIRD_PARTY_PATH}/Vulkan-Headers/registry
                CACHE INTERNAL "" FORCE)
    endif()
else()
    # Vulkan (Static)
    message(STATUS "Build VKW as static-vk mode")
    find_package(Vulkan REQUIRED)
    list(APPEND VKW_INCLUDE ${Vulkan_INCLUDE_DIRS})
    list(APPEND VKW_LIBRARY ${Vulkan_LIBRARY})
    list(APPEND VKW_DEFINE -DVULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=0  # dlopen
                           -DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=0)  # dispatch
endif()

# SPIRV-Headers
set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "")
set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "")
setup_third_party("https://github.com/KhronosGroup/SPIRV-Headers"
                  "2e1bc6dcd4ff49ed996dc0174d1aec37c2e9ab59" TRUE
                  ${VKW_THIRD_PARTY_PATH})
list(APPEND VKW_INCLUDE ${VKW_THIRD_PARTY_PATH}/SPIRV-Headers/include)

# SPIRV-Tools
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "")
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "")
# set(SPIRV_CHECK_CONTEXT ON CACHE BOOL "")
setup_third_party("https://github.com/KhronosGroup/SPIRV-Tools"
                  "ec1bc3e2e5678f2c72c0d3b196780a17df637d60" TRUE
                  ${VKW_THIRD_PARTY_PATH})
list(APPEND VKW_INCLUDE ${VKW_THIRD_PARTY_PATH}/SPIRV-Tools/include)
list(APPEND VKW_LIBRARY SPIRV-Tools-opt)  # For "ENABLE_OPT=1" in GLSLang

# GLSLang
set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "")
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "")
set(ENABLE_HLSL OFF CACHE BOOL "")
set(SKIP_GLSLANG_INSTALL ON CACHE BOOL "")
setup_third_party("https://github.com/KhronosGroup/glslang.git"
                  "6bdcb4be344d7903bd92fd464e496c3199b91484" TRUE
                  ${VKW_THIRD_PARTY_PATH})
list(APPEND VKW_INCLUDE ${VKW_THIRD_PARTY_PATH}/glslang)
list(APPEND VKW_LIBRARY glslang SPIRV glslang-default-resource-limits
                        OGLCompiler)
list(APPEND VKW_DEFINE -DENABLE_OPT=1)  # Defined internally

# Validation Layer
if (ANDROID)
    # Copy library files. Link is not needed because VulkanHpp loads dynamically
    set(VALILAYER_MID_PATH sources/third_party/vulkan/src/build-android/jniLibs)
    file(COPY ${ANDROID_NDK}/${VALILAYER_MID_PATH}/${ANDROID_ABI}
         DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/android/app/src/main/jniLibs/)
endif()

# GLFW
if (VKW_USE_INTERNAL_GLFW)
    message(STATUS "With VKW Internal GLFW")
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
    set(GLFW_INSTALL OFF CACHE BOOL "")
    setup_third_party("https://github.com/glfw/glfw"
                      "114776a24605418e6d719d2f30141e351e93c6e0" TRUE
                      ${VKW_THIRD_PARTY_PATH})
    list(APPEND VKW_INCLUDE ${VKW_THIRD_PARTY_PATH}/glfw/include)
    list(APPEND VKW_LIBRARY glfw)
endif()

if (VKW_BUILD_EXAMPLES)
    message(STATUS "With VKW Examples")

    # GLM
    setup_third_party("https://github.com/g-truc/glm"
                      "66062497b104ca7c297321bd0e970869b1e6ece5" FALSE
                      ${VKW_THIRD_PARTY_PATH})
    list(APPEND VKW_INCLUDE ${VKW_THIRD_PARTY_PATH}/glm)

    # TinyObjLoader
    setup_third_party("https://github.com/syoyo/tinyobjloader"
                      "0ed6c38f20c63b996fbb9fa949569b2acb213a3d" FALSE
                      ${VKW_THIRD_PARTY_PATH})
    add_library(tinyobjloader
        ${VKW_THIRD_PARTY_PATH}/tinyobjloader/tiny_obj_loader.cc
    )
    list(APPEND VKW_LIBRARY tinyobjloader)

    # STB
    setup_third_party("https://github.com/nothings/stb"
                      "c9064e317699d2e495f36ba4f9ac037e88ee371a" FALSE
                      ${VKW_THIRD_PARTY_PATH})
    add_library(stb
        ${CMAKE_CURRENT_SOURCE_DIR}/example/utils/stb_impl.cpp
    )
    setup_target(stb "${VKW_INCLUDE}" "${VKW_LIBRARY}" FALSE)
    list(APPEND VKW_LIBRARY stb)
endif()

# ------------------------------------------------------------------------------
# ----------------------------- Internal Libraries -----------------------------
# ------------------------------------------------------------------------------
# VKW
list(APPEND VKW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(vkw
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vkw.cpp
)
setup_target(vkw "${VKW_INCLUDE}" "${VKW_LIBRARY}" TRUE)
add_definitions(${VKW_DEFINE})
list(APPEND VKW_LIBRARY vkw)

# ------------------------------------------------------------------------------
# ----------------------------- Pass to the parent -----------------------------
# ------------------------------------------------------------------------------
if (HAS_PARENT)
    set(VKW_INCLUDE ${VKW_INCLUDE} PARENT_SCOPE)
    set(VKW_LIBRARY ${VKW_LIBRARY} PARENT_SCOPE)
    set(VKW_DEFINE ${VKW_DEFINE} PARENT_SCOPE)
endif()

# ------------------------------------------------------------------------------
# ---------------------------------- Examples ----------------------------------
# ------------------------------------------------------------------------------
if (VKW_BUILD_EXAMPLES)
    # Example application sources
    set(EXAMPLE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/example/01_rotate_box/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/02_load_obj/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/03_load_obj_many/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/04_deferred_shading/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/05_instancing/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/06_raster_vtx_id/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/07_image_transfer/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/08_comp_shader/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/09_inverse_uv/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/10_glsl_optim/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/11_img_buf/app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/example/12_comp_shader_atomic_float/app.cpp
    )

    if (ANDROID)
        # ----------------------------------------------------------------------
        # --------------------------------- PC ---------------------------------
        # ----------------------------------------------------------------------
        find_library(log-lib log)
        find_library(android-lib android)
        list(APPEND VKW_LIBRARY ${log-lib} ${android-lib})

        # Android example
        add_library(native-lib SHARED
            ${EXAMPLE_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/android/app/src/main/cpp/native-lib.cpp)
        setup_target(native-lib "${VKW_INCLUDE}" "${VKW_LIBRARY}" TRUE)
    else()
        # ----------------------------------------------------------------------
        # ------------------------------ Android -------------------------------
        # ----------------------------------------------------------------------
        # PC example
        add_executable(main
            ${EXAMPLE_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/pc/main.cpp)
        setup_target(main "${VKW_INCLUDE}" "${VKW_LIBRARY}" TRUE)
    endif()
endif()
message(STATUS "--------------------------------------------------------------")
message(STATUS "--------------------------- End VKW --------------------------")
message(STATUS "--------------------------------------------------------------")
