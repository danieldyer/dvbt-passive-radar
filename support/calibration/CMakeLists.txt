add_executable(calibration_archive_probe EXCLUDE_FROM_ALL archive_probe.cpp)
target_link_libraries(calibration_archive_probe PRIVATE radar_dsp)
target_compile_options(calibration_archive_probe PRIVATE -Wall -Wextra -Wpedantic)

add_executable(geographic_profile_export EXCLUDE_FROM_ALL geographic_profile_export.cpp)
target_link_libraries(geographic_profile_export PRIVATE radar_dsp)
target_compile_options(geographic_profile_export PRIVATE -Wall -Wextra -Wpedantic)

add_executable(finite_reference_export EXCLUDE_FROM_ALL finite_reference_export.cpp)
target_link_libraries(finite_reference_export PRIVATE radar_dsp)
target_compile_options(finite_reference_export PRIVATE -Wall -Wextra -Wpedantic)

add_executable(waveform_kernel_export EXCLUDE_FROM_ALL waveform_kernel_export.cpp)
target_link_libraries(waveform_kernel_export PRIVATE radar_dsp)
target_compile_options(waveform_kernel_export PRIVATE -Wall -Wextra -Wpedantic)

set(backprojection_header "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/geographic_backprojection.inc")
add_custom_command(OUTPUT "${backprojection_header}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
    COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -mfmt=c -O
        "${CMAKE_CURRENT_SOURCE_DIR}/geographic_backprojection.comp" -o "${backprojection_header}"
    DEPENDS geographic_backprojection.comp VERBATIM)
add_executable(geographic_backprojection EXCLUDE_FROM_ALL geographic_backprojection.cpp "${backprojection_header}")
target_link_libraries(geographic_backprojection PRIVATE radar_frontend)
target_include_directories(geographic_backprojection PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
target_compile_options(geographic_backprojection PRIVATE -Wall -Wextra -Wpedantic)
