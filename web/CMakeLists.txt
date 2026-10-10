cmake_minimum_required(VERSION 3.28)
project(radar_viewer LANGUAGES CXX)
if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "Configure with emcmake cmake -S web -B build/web")
endif()
include(FetchContent)
FetchContent_Declare(imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.9b.tar.gz
    URL_HASH SHA256=8e1bbc76c71d74fef2fb85db7e7ca8eba13d6a86623c54992b60162db554ffdb
    SOURCE_SUBDIR unused)
FetchContent_MakeAvailable(imgui)
add_executable(index main.cpp ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_compile_features(index PRIVATE cxx_std_23)
target_include_directories(index PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_compile_options(index PRIVATE -O3 -sUSE_SDL=2 -DIMGUI_IMPL_OPENGL_ES3)
target_link_options(index PRIVATE -O3 -sUSE_SDL=2 -sUSE_WEBGL2=1 -sALLOW_MEMORY_GROWTH=1
    -sEXPORTED_FUNCTIONS=['_main','_receive_frame','_malloc','_free']
    -sEXPORTED_RUNTIME_METHODS=['HEAPU8'] --shell-file ${CMAKE_CURRENT_SOURCE_DIR}/shell.html)
target_link_options(index PRIVATE --embed-file ${imgui_SOURCE_DIR}/misc/fonts/Roboto-Medium.ttf@/font.ttf)
set_target_properties(index PROPERTIES SUFFIX .html)
