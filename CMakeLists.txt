cmake_minimum_required(VERSION 3.28)
project(dvbt_passive_radar VERSION 0.1.0 LANGUAGES CXX)

if(DEFINED ENV{VULKAN_SDK})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
endif()

find_package(Vulkan 1.2 REQUIRED COMPONENTS glslc)
find_package(glslang CONFIG REQUIRED)

include(FetchContent)
# VkFFT v1.3.4, as used by sar_processing. Fetch only its headers.
FetchContent_Declare(vkfft
    URL https://github.com/DTolm/VkFFT/archive/066a17c17068c0f11c9298d848c2976c71fad1c1.tar.gz
    URL_HASH SHA256=9b6a03cce4a9aab758b4bb57357cb5e2cc15f4ab6cd00f2e35c37edf03ec2424
    SOURCE_SUBDIR vkFFT
)
FetchContent_MakeAvailable(vkfft)

# VkFFT includes glslang's C interface without the glslang/Include prefix.
get_target_property(glslang_includes glslang::glslang INTERFACE_INCLUDE_DIRECTORIES)
find_path(VKFFT_GLSLANG_INCLUDE_DIR glslang_c_interface.h
    HINTS ${glslang_includes} ${Vulkan_INCLUDE_DIRS}
    PATH_SUFFIXES glslang/Include REQUIRED
)
add_library(vkfft_headers INTERFACE)
target_include_directories(vkfft_headers SYSTEM INTERFACE
    "${vkfft_SOURCE_DIR}/vkFFT" "${VKFFT_GLSLANG_INCLUDE_DIR}"
)
target_compile_definitions(vkfft_headers INTERFACE VKFFT_BACKEND=0)
target_link_libraries(vkfft_headers INTERFACE
    Vulkan::Vulkan glslang::glslang glslang::SPIRV
)

include(CTest)
include(cmake/dvbt.cmake)
