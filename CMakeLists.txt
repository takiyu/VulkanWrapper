cmake_minimum_required(VERSION 3.8.2)

# ------------------------------------------------------------------------------
# -------------------------------- Check parent --------------------------------
# ------------------------------------------------------------------------------
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    message(STATUS "Build VKW as subdirectory mode")
else()
    message(STATUS "Build VKW as independent mode")
endif()

# ------------------------------------------------------------------------------
# ----------------------------------- Options ----------------------------------
# ------------------------------------------------------------------------------
set(VKW_BUILD_EXAMPLES ON CACHE BOOL "Build examples")
set(VKW_LINK_DYNAMIC_VK ON CACHE BOOL "Dynamic link for vulkan.")

# ------------------------------------------------------------------------------
# ----------------------------------- Common -----------------------------------
# ------------------------------------------------------------------------------
project(vulkan_practice CXX C)

set(CMAKE_CXX_STANDARD 14)  # C++ 14

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(LINK_TYPE STATIC)
else()
    set(LINK_TYPE SHARED)
endif()

# Print make commands for debug
# set(CMAKE_VERBOSE_MAKEFILE 1)

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Export `compile_commands.json`
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers)
find_package(Sanitizers) # Address sanitizer (-DSANITIZE_ADDRESS=ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Warning options
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warning_options "-Wall -Wextra -Wconversion")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(warning_options "-Wall -Wextra -Wcast-align -Wcast-qual \
                         -Wctor-dtor-privacy -Wdisabled-optimization \
                         -Wformat=2 -Winit-self \
                         -Wmissing-declarations -Wmissing-include-dirs \
                         -Woverloaded-virtual -Wredundant-decls -Wshadow \
                         -Wsign-conversion -Wsign-promo \
                         -Wstrict-overflow=5 -Wundef -Wno-unknown-pragmas")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(warning_options "--pedantic -Wall -Wextra -Wcast-align -Wcast-qual \
                         -Wctor-dtor-privacy -Wdisabled-optimization \
                         -Wformat=2 -Winit-self -Wlogical-op \
                         -Wmissing-declarations -Wmissing-include-dirs \
                         -Wnoexcept -Woverloaded-virtual \
                         -Wredundant-decls -Wshadow -Wsign-conversion \
                         -Wsign-promo -Wstrict-null-sentinel \
                         -Wstrict-overflow=5 -Wundef -Wno-unknown-pragmas")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warning_options "/W4")
else()
    message(WARNING "Unsupported compiler for warning options")
    message ("CMAKE_CXX_COMPILER_ID is ${CMAKE_CXX_COMPILER_ID}")
endif()

# Utility function to setup a target (include, link, warning, sanitizer)
function(setup_target target includes libs)
    target_include_directories(${target} PUBLIC ${includes})
    target_link_libraries(${target} ${libs})
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${warning_options})
endfunction(setup_target)

# Utility function to setup a target simply (include, link)
function(setup_target_simple target includes libs)
    target_include_directories(${target} PUBLIC ${includes})
    target_link_libraries(${target} ${libs})
endfunction(setup_target_simple)

# ------------------------------------------------------------------------------
# --------------------------- Third Party Libraries ----------------------------
# ------------------------------------------------------------------------------
list(APPEND VKW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if (VKW_LINK_DYNAMIC_VK)
    # Vulkan (Dynamic)
    message(STATUS "Build VKW as dynacmic-vk mode")
    list(APPEND VKW_INCLUDE
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-Headers/include)
else()
    # Vulkan (Static)
    message(STATUS "Build VKW as static-vk mode")
    find_package(Vulkan REQUIRED)
    list(APPEND VKW_INCLUDE ${Vulkan_INCLUDE_DIRS})
    list(APPEND VKW_LIBRARY ${Vulkan_LIBRARY})
endif()

# SPIRV-Headers
set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "")
set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/SPIRV-Headers)
list(APPEND VKW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/SPIRV-Headers/include)
list(APPEND VKW_LIBRARY SPIRV-Tools-opt)

# SPIRV-Tools
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "")
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "")
# set(SPIRV_CHECK_CONTEXT ON CACHE BOOL "")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/SPIRV-Tools)
list(APPEND VKW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/SPIRV-Tools/include)
list(APPEND VKW_LIBRARY SPIRV-Tools-opt)

# GLSLang
set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "")
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "")
set(ENABLE_HLSL OFF CACHE BOOL "")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glslang)
list(APPEND VKW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glslang)
list(APPEND VKW_LIBRARY glslang SPIRV glslang-default-resource-limits
                        OGLCompiler)

# GLM
list(APPEND VKW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm)

# Validation Layer
if (ANDROID)
    # Copy library files. Link is not needed because VulkanHpp loads dynamically
    set(VALILAYER_MID_PATH sources/third_party/vulkan/src/build-android/jniLibs)
    file(COPY ${ANDROID_NDK}/${VALILAYER_MID_PATH}/${ANDROID_ABI}
         DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/android/app/src/main/jniLibs/)
endif()

# GLFW3
if (NOT ANDROID)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
    set(GLFW_INSTALL OFF CACHE BOOL "")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw)
    list(APPEND VKW_INCLUDE
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include)
    list(APPEND VKW_LIBRARY glfw)
endif()

if (VKW_BUILD_EXAMPLES)
    # TinyObjLoader
    add_library(tinyobjloader
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyobjloader/tiny_obj_loader.cc
    )
    list(APPEND VKW_LIBRARY tinyobjloader)

    # STB
    add_library(stb
        ${CMAKE_CURRENT_SOURCE_DIR}/example/utils/stb_impl.cpp
    )
    setup_target_simple(stb "${VKW_INCLUDE}" "${VKW_LIBRARY}")
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
setup_target(vkw "${VKW_INCLUDE}" "${VKW_LIBRARY}")
list(APPEND VKW_LIBRARY vkw)

# ------------------------------------------------------------------------------
# ----------------------------- Pass to the parent -----------------------------
# ------------------------------------------------------------------------------
if(HAS_PARENT)
    set(VKW_INCLUDE ${VKW_INCLUDE} PARENT_SCOPE)
    set(VKW_LIBRARY ${VKW_LIBRARY} PARENT_SCOPE)
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
        setup_target(native-lib "${VKW_INCLUDE}" "${VKW_LIBRARY}")
    else()
        # ----------------------------------------------------------------------
        # ------------------------------ Android -------------------------------
        # ----------------------------------------------------------------------
        # PC example
        add_executable(main
            ${EXAMPLE_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/pc/main.cpp)
        setup_target(main "${VKW_INCLUDE}" "${VKW_LIBRARY}")
    endif()
endif()
