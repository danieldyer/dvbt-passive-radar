# GNU Radio's runtime/FFT/FEC are required; no Python runtime or GUI is used.
list(PREPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/build/deps")
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
find_package(Gnuradio 3.10 REQUIRED COMPONENTS runtime blocks fft filter fec dtv)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_path(FFTW_INCLUDE_DIR fftw3.h REQUIRED)
find_library(RADAR_FFTW_LIBRARY fftw3f HINTS "${PROJECT_SOURCE_DIR}/build/deps/lib" REQUIRED)

FetchContent_Declare(dl5eu
    URL https://github.com/dl5eu/gr-dl5eu/archive/d6ab8c12c1283f144b17e4c0ca2b6727a084339a.tar.gz
    URL_HASH SHA256=90e3b00bfe94c096ea35a502d3266ae5b5c55365e7b32dca67f2127adea551fe
    SOURCE_SUBDIR unused)
FetchContent_MakeAvailable(dl5eu)
set(sync_dir "${CMAKE_CURRENT_BINARY_DIR}/instrumented_sync")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/patch_sync.py"
    "${dl5eu_SOURCE_DIR}" "${sync_dir}" COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/scripts/patch_sync.py")
add_library(dvbt_sync STATIC
    "${sync_dir}/dvbt_configure.cc" "${sync_dir}/dvbt_pilot_gen.cc"
    "${sync_dir}/dvbt_ofdm_synchronization_impl.cc")
target_include_directories(dvbt_sync SYSTEM PUBLIC "${sync_dir}" "${dl5eu_SOURCE_DIR}/include")
target_include_directories(dvbt_sync PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(dvbt_sync PRIVATE cxx_std_23)
target_link_libraries(dvbt_sync PUBLIC gnuradio::gnuradio-runtime gnuradio::gnuradio-fft gnuradio::gnuradio-filter)
set(frontend_headers)
foreach(shader channel_fold channel_unpack resample spectral_boundary spectral_synth)
    set(header "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/${shader}.inc")
    add_custom_command(OUTPUT "${header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -mfmt=c
            "$<IF:$<CONFIG:Debug>,-g,-O>" "${PROJECT_SOURCE_DIR}/shaders/${shader}.comp" -o "${header}"
        DEPENDS "shaders/${shader}.comp" VERBATIM)
    list(APPEND frontend_headers "${header}")
endforeach()
set(pipeline_headers)
foreach(shader ofdm_window ofdm_pilots ofdm_equalize ofdm_cp ofdm_frequency ofdm_rotate ofdm_tps stream_copy reencode remap carrier_copy response_reduce response_fit response_apply noise_modes noise_variance matched_pack matched_pad matched_product matched_extract focus_mean focus_pack focus_shear focus_extract)
    set(header "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/${shader}.inc")
    add_custom_command(OUTPUT "${header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -mfmt=c
            "$<IF:$<CONFIG:Debug>,-g,-O>" "${PROJECT_SOURCE_DIR}/shaders/${shader}.comp" -o "${header}"
        DEPENDS "shaders/${shader}.comp" VERBATIM)
    list(APPEND pipeline_headers "${header}")
endforeach()
add_library(radar_frontend src/dsp.cpp src/gpu_frontend.cpp src/spectral_clock.cpp src/sample_clock.cpp ${frontend_headers})
target_compile_features(radar_frontend PUBLIC cxx_std_23)
set_target_properties(radar_frontend PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(radar_frontend PUBLIC src PRIVATE "${FFTW_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
target_link_libraries(radar_frontend PUBLIC "${RADAR_FFTW_LIBRARY}" Vulkan::Vulkan Volk::volk PRIVATE vkfft_headers)
target_compile_options(radar_frontend PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(dvbt_sync PUBLIC radar_frontend)
set(residual_header "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/residual.inc")
add_custom_command(OUTPUT "${residual_header}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
    COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -mfmt=c
        "$<IF:$<CONFIG:Debug>,-g,-O>" "${PROJECT_SOURCE_DIR}/shaders/residual.comp" -o "${residual_header}"
    DEPENDS shaders/residual.comp VERBATIM)
add_library(radar_dsp src/modem.cpp src/fec.cpp src/cpu_fec.cpp src/gpu_reconstruction.cpp src/gpu_frames.cpp src/frame_decoder.cpp src/gpu_ofdm.cpp src/gpu_receiver.cpp src/gpu_response.cpp src/gpu_pipeline.cpp src/gpu_noise.cpp src/matched_profiles.cpp src/gpu_focus.cpp src/sar_handoff.cpp src/pipeline.cpp src/gpu_residual.cpp src/range_doppler.cpp "${residual_header}" ${pipeline_headers})
target_compile_features(radar_dsp PUBLIC cxx_std_23)
set_target_properties(radar_dsp PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(radar_dsp PUBLIC src PRIVATE "${FFTW_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
target_link_libraries(radar_dsp PUBLIC dvbt_sync "${RADAR_FFTW_LIBRARY}" Vulkan::Vulkan PRIVATE vkfft_headers)
target_compile_options(radar_dsp PRIVATE -Wall -Wextra -Wpedantic)
add_executable(dvbt_reconstruct src/reconstruct_main.cpp)
target_link_libraries(dvbt_reconstruct PRIVATE radar_dsp)
target_sources(radar_dsp PRIVATE src/matched_detector.cpp src/stationary_background.cpp)
add_executable(dvbt_detect src/detect_main.cpp)
target_link_libraries(dvbt_detect PRIVATE radar_dsp)
add_executable(dvbt_sar src/sar_main.cpp)
target_link_libraries(dvbt_sar PRIVATE radar_dsp)
add_subdirectory(support/validation EXCLUDE_FROM_ALL)
if(BUILD_TESTING)
    add_executable(matched_tests tests/matched_tests.cpp)
    target_link_libraries(matched_tests PRIVATE radar_dsp)
    add_test(NAME matched_profiles_gpu COMMAND matched_tests)
    set_tests_properties(matched_profiles_gpu PROPERTIES TIMEOUT 180 LABELS gpu
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    set(command_header "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/command_chain.inc")
    add_custom_command(OUTPUT "${command_header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -mfmt=c
            "$<IF:$<CONFIG:Debug>,-g,-O>" "${PROJECT_SOURCE_DIR}/tests/shaders/command_chain.comp" -o "${command_header}"
        DEPENDS tests/shaders/command_chain.comp VERBATIM)
    add_executable(command_tests tests/command_tests.cpp "${command_header}")
    target_include_directories(command_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
    target_link_libraries(command_tests PRIVATE radar_frontend)
    add_test(NAME command_dependencies_gpu COMMAND command_tests)
    set_tests_properties(command_dependencies_gpu PROPERTIES TIMEOUT 60 LABELS gpu
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(sar_tests tests/sar_tests.cpp)
    target_link_libraries(sar_tests PRIVATE radar_dsp)
    add_test(NAME sar_coordinates_cpu COMMAND sar_tests)
    add_test(NAME sar_multichannel_gpu COMMAND sar_tests --gpu)
    add_test(NAME sar_bounded_workers_gpu COMMAND sar_tests --gpu-bounded)
    add_test(NAME sar_consumer_failure_gpu COMMAND sar_tests --gpu-failure)
    add_test(NAME sar_mixed_mcs_gpu COMMAND sar_tests --gpu-mixed)
    set_tests_properties(sar_coordinates_cpu PROPERTIES TIMEOUT 60 LABELS cpu)
    set_tests_properties(sar_multichannel_gpu sar_bounded_workers_gpu sar_consumer_failure_gpu sar_mixed_mcs_gpu PROPERTIES TIMEOUT 300 LABELS gpu
        ENVIRONMENT "TMP=${PROJECT_BINARY_DIR};XDG_CACHE_HOME=${PROJECT_BINARY_DIR}/cache;XDG_CONFIG_HOME=${PROJECT_BINARY_DIR}/config"
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    set_tests_properties(sar_mixed_mcs_gpu PROPERTIES TIMEOUT 600)
    add_executable(response_tests tests/response_tests.cpp)
    target_link_libraries(response_tests PRIVATE radar_dsp)
    add_test(NAME response_gpu COMMAND response_tests)
    set_tests_properties(response_gpu PROPERTIES TIMEOUT 180 LABELS gpu
        ENVIRONMENT "TMP=${PROJECT_BINARY_DIR};XDG_CACHE_HOME=${PROJECT_BINARY_DIR}/cache;XDG_CONFIG_HOME=${PROJECT_BINARY_DIR}/config"
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(reconstruction_tests tests/reconstruction_tests.cpp)
    target_link_libraries(reconstruction_tests PRIVATE radar_dsp)
    add_test(NAME remodulation_gpu COMMAND reconstruction_tests)
    set_tests_properties(remodulation_gpu PROPERTIES TIMEOUT 180 LABELS gpu
        ENVIRONMENT "TMP=${PROJECT_BINARY_DIR};XDG_CACHE_HOME=${PROJECT_BINARY_DIR}/cache;XDG_CONFIG_HOME=${PROJECT_BINARY_DIR}/config"
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(receiver_tests tests/receiver_tests.cpp)
    target_link_libraries(receiver_tests PRIVATE radar_dsp)
    add_test(NAME receiver_gpu COMMAND receiver_tests)
    set_tests_properties(receiver_gpu PROPERTIES TIMEOUT 180 LABELS gpu
        ENVIRONMENT "TMP=${PROJECT_BINARY_DIR};XDG_CACHE_HOME=${PROJECT_BINARY_DIR}/cache;XDG_CONFIG_HOME=${PROJECT_BINARY_DIR}/config"
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(ofdm_tests tests/ofdm_tests.cpp)
    target_link_libraries(ofdm_tests PRIVATE radar_dsp)
    add_test(NAME ofdm_gpu COMMAND ofdm_tests)
    set_tests_properties(ofdm_gpu PROPERTIES TIMEOUT 180 LABELS gpu
        ENVIRONMENT "TMP=${PROJECT_BINARY_DIR};XDG_CACHE_HOME=${PROJECT_BINARY_DIR}/cache;XDG_CONFIG_HOME=${PROJECT_BINARY_DIR}/config"
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(fec_tests tests/fec_tests.cpp)
    target_link_libraries(fec_tests PRIVATE radar_dsp gnuradio::gnuradio-fec)
    add_test(NAME fec_cpu COMMAND fec_tests)
    set_tests_properties(fec_cpu PROPERTIES TIMEOUT 180 LABELS cpu)
    add_executable(pipeline_benchmarks EXCLUDE_FROM_ALL tests/pipeline_benchmarks.cpp)
    target_link_libraries(pipeline_benchmarks PRIVATE radar_dsp)
    add_executable(fec_pipeline_benchmark EXCLUDE_FROM_ALL tests/fec_pipeline_benchmark.cpp)
    target_link_libraries(fec_pipeline_benchmark PRIVATE radar_dsp)
    add_executable(fec_benchmarks tests/fec_benchmarks.cpp)
    target_link_libraries(fec_benchmarks PRIVATE radar_dsp)
    add_executable(common_clock_tests tests/common_clock_tests.cpp)
    target_link_libraries(common_clock_tests PRIVATE radar_frontend)
    add_test(NAME common_clock_cpu COMMAND common_clock_tests)
    add_test(NAME common_clock_gpu COMMAND common_clock_tests --gpu)
    set_tests_properties(common_clock_cpu PROPERTIES TIMEOUT 60 LABELS cpu)
    set_tests_properties(common_clock_gpu PROPERTIES TIMEOUT 180 LABELS gpu ENVIRONMENT "TMP=${PROJECT_BINARY_DIR}"
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(spectral_clock_tests tests/spectral_clock_tests.cpp)
    target_link_libraries(spectral_clock_tests PRIVATE radar_frontend)
    add_test(NAME spectral_clock_cpu COMMAND spectral_clock_tests)
    add_test(NAME spectral_clock_gpu COMMAND spectral_clock_tests --gpu)
    set_tests_properties(spectral_clock_cpu PROPERTIES TIMEOUT 60 LABELS cpu)
    set_tests_properties(spectral_clock_gpu PROPERTIES TIMEOUT 180 LABELS gpu
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(frontend_tests tests/frontend_tests.cpp)
    target_link_libraries(frontend_tests PRIVATE radar_frontend)
    add_test(NAME frontend_gpu COMMAND frontend_tests)
    set_tests_properties(frontend_gpu PROPERTIES TIMEOUT 180 LABELS gpu
        ENVIRONMENT "TMP=${PROJECT_BINARY_DIR}" FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(frontend_benchmarks EXCLUDE_FROM_ALL tests/frontend_benchmarks.cpp)
    target_link_libraries(frontend_benchmarks PRIVATE radar_frontend)
    set(raw_clock_reference "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/raw_clock_reference.inc")
    add_custom_command(OUTPUT "${raw_clock_reference}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -Werror -mfmt=c
            "$<IF:$<CONFIG:Debug>,-g,-O>" "${PROJECT_SOURCE_DIR}/tests/shaders/raw_clock_reference.comp" -o "${raw_clock_reference}"
        DEPENDS tests/shaders/raw_clock_reference.comp VERBATIM)
    add_executable(resampling_benchmarks EXCLUDE_FROM_ALL tests/resampling_benchmarks.cpp "${raw_clock_reference}")
    target_include_directories(resampling_benchmarks PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
    target_link_libraries(resampling_benchmarks PRIVATE radar_frontend)
    # Explicit target: benchmarks are opt-in and are not pass/fail CTest cases.
    add_executable(radar_benchmarks EXCLUDE_FROM_ALL tests/benchmarks.cpp)
    target_link_libraries(radar_benchmarks PRIVATE radar_dsp vkfft_headers)
    add_executable(range_doppler_tests tests/range_doppler_tests.cpp)
    target_link_libraries(range_doppler_tests PRIVATE radar_dsp)
    add_test(NAME range_doppler_cpu COMMAND range_doppler_tests)
    set_tests_properties(range_doppler_cpu PROPERTIES TIMEOUT 60 LABELS cpu ENVIRONMENT "TMP=${PROJECT_BINARY_DIR}")
    add_executable(phase_tests tests/phase_tests.cpp)
    target_link_libraries(phase_tests PRIVATE radar_dsp)
    add_test(NAME phase_coordinates_cpu COMMAND phase_tests)
    set_tests_properties(phase_coordinates_cpu PROPERTIES TIMEOUT 180 LABELS cpu)
    add_test(NAME phase_coordinates_gpu COMMAND phase_tests --gpu)
    add_test(NAME receiver_phase_gpu COMMAND phase_tests --receiver)
    add_test(NAME pipeline_phase_gpu COMMAND phase_tests --pipeline)
    set_tests_properties(phase_coordinates_gpu receiver_phase_gpu pipeline_phase_gpu PROPERTIES TIMEOUT 180 LABELS gpu
        ENVIRONMENT "TMP=${PROJECT_BINARY_DIR};XDG_CACHE_HOME=${PROJECT_BINARY_DIR}/cache;XDG_CONFIG_HOME=${PROJECT_BINARY_DIR}/config"
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    add_executable(radar_tests tests/dsp_tests.cpp)
    target_link_libraries(radar_tests PRIVATE radar_dsp gnuradio::gnuradio-dtv gnuradio::gnuradio-blocks)
    add_test(NAME reconstruction_cpu COMMAND radar_tests)
    set_tests_properties(reconstruction_cpu PROPERTIES TIMEOUT 180 LABELS cpu)
    add_test(NAME residual_gpu COMMAND radar_tests --gpu)
    set_tests_properties(residual_gpu PROPERTIES TIMEOUT 60 LABELS gpu
        FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    set_tests_properties(reconstruction_cpu phase_coordinates_cpu residual_gpu PROPERTIES ENVIRONMENT
        "TMP=${PROJECT_BINARY_DIR};XDG_CACHE_HOME=${PROJECT_BINARY_DIR}/cache;XDG_CONFIG_HOME=${PROJECT_BINARY_DIR}/config")
endif()
