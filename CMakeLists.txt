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

# Compile and embed SPIR-V: the executable does not depend on its working directory.
set(shader_header "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/cross_spectrum.inc")
add_custom_command(OUTPUT "${shader_header}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
    COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -mfmt=c
        "$<IF:$<CONFIG:Debug>,-g,-O>"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/cross_spectrum.comp"
        -o "${shader_header}"
    DEPENDS shaders/cross_spectrum.comp
    COMMENT "Compiling cross_spectrum.comp"
    VERBATIM
)

add_executable(dvbt_passive_radar src/main.cpp "${shader_header}")
target_compile_features(dvbt_passive_radar PRIVATE cxx_std_23)
set_target_properties(dvbt_passive_radar PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(dvbt_passive_radar PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
)
target_link_libraries(dvbt_passive_radar PRIVATE vkfft_headers)
if(MSVC)
    target_compile_options(dvbt_passive_radar PRIVATE /W4 /permissive- /Zc:__cplusplus)
else()
    target_compile_options(dvbt_passive_radar PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
endif()

include(CTest)
option(RADAR_BUILD_PIPELINE "Build recorded-IQ DVB-T reconstruction pipeline" ON)
if(RADAR_BUILD_PIPELINE)
    include(cmake/dvbt.cmake)
endif()

if(BUILD_TESTING)
    add_test(NAME cross_spectrum_8k COMMAND dvbt_passive_radar 8192)
    add_test(NAME cross_spectrum_partial_workgroup COMMAND dvbt_passive_radar 320)
    set_tests_properties(cross_spectrum_8k cross_spectrum_partial_workgroup
        PROPERTIES LABELS gpu TIMEOUT 60
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD"
    )
endif()
