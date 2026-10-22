set(isar_shaders)
foreach(stage backprojection backprojection_update reduce interpolate_profiles focus_patch focus_reduce focus_adjoint focus_adjoint_reduce focus_histories coherent_glrt template_information)
    set(header "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/${stage}.inc")
    add_custom_command(OUTPUT "${header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -mfmt=c -O
            "${CMAKE_CURRENT_SOURCE_DIR}/${stage}.comp" -o "${header}"
        DEPENDS "${stage}.comp" focus_sample.glsl backprojection_body.glsl VERBATIM)
    list(APPEND isar_shaders "${header}")
endforeach()
add_executable(isar_backprojection EXCLUDE_FROM_ALL backprojection.cpp ${isar_shaders})
target_link_libraries(isar_backprojection PRIVATE radar_frontend)
target_include_directories(isar_backprojection PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
target_compile_options(isar_backprojection PRIVATE -Wall -Wextra -Wpedantic)

add_executable(isar_focus_patch EXCLUDE_FROM_ALL focus_patch.cpp ${isar_shaders})
target_link_libraries(isar_focus_patch PRIVATE radar_frontend)
target_include_directories(isar_focus_patch PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
target_compile_options(isar_focus_patch PRIVATE -Wall -Wextra -Wpedantic)

add_executable(isar_coherent_statistics EXCLUDE_FROM_ALL coherent_statistics.cpp ${isar_shaders})
target_link_libraries(isar_coherent_statistics PRIVATE radar_frontend)
target_include_directories(isar_coherent_statistics PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
target_compile_options(isar_coherent_statistics PRIVATE -Wall -Wextra -Wpedantic)

add_executable(isar_template_information EXCLUDE_FROM_ALL template_information.cpp ${isar_shaders})
target_link_libraries(isar_template_information PRIVATE radar_frontend)
target_include_directories(isar_template_information PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
target_compile_options(isar_template_information PRIVATE -Wall -Wextra -Wpedantic)
