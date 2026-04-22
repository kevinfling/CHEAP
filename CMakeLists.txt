cmake_minimum_required(VERSION 3.16)
project(CHEAP VERSION 0.2.0 LANGUAGES C)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build types
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type: Debug | Release | Asan | Ubsan | Tsan" FORCE)
endif()

set(CMAKE_C_FLAGS_DEBUG   "-O0 -g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_ASAN    "-O1 -g -fsanitize=address,undefined")
set(CMAKE_EXE_LINKER_FLAGS_ASAN "-fsanitize=address,undefined")
set(CMAKE_C_FLAGS_UBSAN   "-O1 -g -fsanitize=undefined -fno-sanitize-recover=all")
set(CMAKE_EXE_LINKER_FLAGS_UBSAN "-fsanitize=undefined")
set(CMAKE_C_FLAGS_TSAN    "-O1 -g -fsanitize=thread")
set(CMAKE_EXE_LINKER_FLAGS_TSAN "-fsanitize=thread")

# Mirror sanitizer flags to C++ so cheap_cpp targets inherit them
set(CMAKE_CXX_FLAGS_ASAN  "${CMAKE_C_FLAGS_ASAN}")
set(CMAKE_CXX_FLAGS_UBSAN "${CMAKE_C_FLAGS_UBSAN}")
set(CMAKE_CXX_FLAGS_TSAN  "${CMAKE_C_FLAGS_TSAN}")

# ---------------------------------------------------------------------------
# FFTW3 detection — three-tier fallback
# ---------------------------------------------------------------------------

# Tier 1: pkg-config
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
endif()

# Tier 2: custom FindFFTW3.cmake (covers Homebrew, custom prefix)
if(NOT FFTW3_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    find_package(FFTW3 QUIET)
endif()

# Tier 3: FetchContent — download FFTW 3.3.10 source
if(NOT FFTW3_FOUND)
    message(STATUS "FFTW3 not found locally — downloading via FetchContent")
    include(FetchContent)

    set(ENABLE_FLOAT        OFF CACHE BOOL "" FORCE)
    set(ENABLE_LONG_DOUBLE  OFF CACHE BOOL "" FORCE)
    set(ENABLE_THREADS      OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPENMP       OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS         OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(fftw3
        URL      https://fftw.org/fftw-3.3.10.tar.gz
        URL_HASH SHA256=56c932549852cddcfabd24772ec6a789aa7b625f63d9a7c06b95b3becea8b697)
    FetchContent_MakeAvailable(fftw3)

    if(TARGET fftw3 AND NOT TARGET FFTW3::fftw3)
        add_library(FFTW3::fftw3 ALIAS fftw3)
    endif()
    set(FFTW3_FOUND TRUE)
endif()

# Normalize pkg-config imported target to FFTW3::fftw3
if(TARGET PkgConfig::FFTW3 AND NOT TARGET FFTW3::fftw3)
    add_library(FFTW3::fftw3 ALIAS PkgConfig::FFTW3)
endif()

if(NOT TARGET FFTW3::fftw3)
    message(FATAL_ERROR "FFTW3 could not be found or built.")
endif()

# ---------------------------------------------------------------------------
# CHEAP INTERFACE library
# ---------------------------------------------------------------------------
add_library(cheap INTERFACE)
target_include_directories(cheap INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(cheap INTERFACE
    -Wall -Wextra -Werror -pedantic -march=native)
target_link_libraries(cheap INTERFACE FFTW3::fftw3 m)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_cheap tests/test_cheap.c)
target_link_libraries(test_cheap PRIVATE cheap)
target_compile_definitions(test_cheap PRIVATE _POSIX_C_SOURCE=199309L)
add_test(NAME unit COMMAND test_cheap)
set_tests_properties(unit PROPERTIES TIMEOUT 60)

add_executable(test_simd tests/test_simd.c)
target_link_libraries(test_simd PRIVATE cheap)
target_compile_definitions(test_simd PRIVATE _POSIX_C_SOURCE=199309L)
add_test(NAME simd COMMAND test_simd)
set_tests_properties(simd PROPERTIES TIMEOUT 60)

add_executable(test_cheap_2d tests/test_cheap_2d.c)
target_link_libraries(test_cheap_2d PRIVATE cheap)
target_compile_definitions(test_cheap_2d PRIVATE _POSIX_C_SOURCE=199309L)
add_test(NAME unit_2d COMMAND test_cheap_2d)
set_tests_properties(unit_2d PROPERTIES TIMEOUT 60)

add_executable(test_cheap_3d tests/test_cheap_3d.c)
target_link_libraries(test_cheap_3d PRIVATE cheap)
target_compile_definitions(test_cheap_3d PRIVATE _POSIX_C_SOURCE=199309L)
add_test(NAME unit_3d COMMAND test_cheap_3d)
set_tests_properties(unit_3d PROPERTIES TIMEOUT 120)

# ---------------------------------------------------------------------------
# Benchmarks (built but NOT registered with CTest)
# ---------------------------------------------------------------------------
# Native benchmark (SIMD path via CHEAP_SIMD_* macros — NEON/AVX2/scalar fallback)
add_executable(bench_cheap benchmarks/bench_cheap.c)
target_link_libraries(bench_cheap PRIVATE cheap)
target_compile_definitions(bench_cheap PRIVATE _POSIX_C_SOURCE=199309L)

# Scalar twin: identical code compiled with CHEAP_SIMD_DISABLE so the pointwise-mul
# and scale-copy helpers fall back to their scalar implementations. Same optimizer
# flags as the native build so we're measuring SIMD vs scalar, not opt level.
add_executable(bench_cheap_scalar benchmarks/bench_cheap.c)
target_link_libraries(bench_cheap_scalar PRIVATE cheap)
target_compile_definitions(bench_cheap_scalar PRIVATE _POSIX_C_SOURCE=199309L CHEAP_SIMD_DISABLE=1)

# -ffast-math only outside sanitizer builds (UBSan flags inf/nan use under fast-math).
# Clang ≥ 22 also warns on isfinite()/INFINITY under -ffast-math; silence since we
# still want the isfinite guards to compile (they short-circuit on finite inputs).
if(NOT CMAKE_BUILD_TYPE MATCHES "^(Asan|Ubsan|Tsan)$")
    target_compile_options(bench_cheap PRIVATE -O3 -ffast-math -fopenmp-simd)
    target_compile_options(bench_cheap_scalar PRIVATE -O3 -ffast-math -fopenmp-simd)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        target_compile_options(bench_cheap PRIVATE -Wno-nan-infinity-disabled)
        target_compile_options(bench_cheap_scalar PRIVATE -Wno-nan-infinity-disabled)
    endif()
else()
    target_compile_options(bench_cheap PRIVATE -fopenmp-simd)
    target_compile_options(bench_cheap_scalar PRIVATE -fopenmp-simd)
endif()

# Statistical benchmark (FFTW_PATIENT with repeated trials)
add_executable(bench_cheap_stats benchmarks/bench_cheap_stats.c)
target_link_libraries(bench_cheap_stats PRIVATE cheap)
target_compile_definitions(bench_cheap_stats PRIVATE _POSIX_C_SOURCE=199309L)

# ---------------------------------------------------------------------------
# C Examples (built but NOT registered with CTest)
# ---------------------------------------------------------------------------
set(CHEAP_EXAMPLES_C
    gp_regression
    toeplitz_solve
    optimal_transport
    lqr_mpc
    fractional_diff
    poisson_solver
    ns_dissipation
    online_filter
    wiener_denoise
    rmt_denoise
    poisson_solver_2d
    poisson_solver_3d
    wiener_denoise_2d
    ns_dissipation_2d
    ns_dissipation_3d
    gp_regression_2d
    toeplitz_solve_2d
)

foreach(example ${CHEAP_EXAMPLES_C})
    add_executable(${example} examples/c/${example}.c)
    target_link_libraries(${example} PRIVATE cheap)
    target_compile_definitions(${example} PRIVATE _POSIX_C_SOURCE=199309L)
endforeach()

# ---------------------------------------------------------------------------
# CHEAP C++ wrapper (header-only, C++17)
# ---------------------------------------------------------------------------
option(CHEAP_BUILD_CPP "Build C++ wrapper targets" ON)

if(CHEAP_BUILD_CPP)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    add_library(cheap_cpp INTERFACE)
    target_include_directories(cheap_cpp INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_options(cheap_cpp INTERFACE
        -Wall -Wextra -Werror -march=native)
    target_link_libraries(cheap_cpp INTERFACE FFTW3::fftw3 m)

    # C++ tests
    add_executable(test_cheap_cpp tests/test_cheap_cpp.cpp)
    target_link_libraries(test_cheap_cpp PRIVATE cheap_cpp)
    target_compile_definitions(test_cheap_cpp PRIVATE _POSIX_C_SOURCE=199309L)
    add_test(NAME unit_cpp COMMAND test_cheap_cpp)
    set_tests_properties(unit_cpp PROPERTIES TIMEOUT 60)

    # C++ benchmarks
    add_executable(bench_cheap_cpp benchmarks/bench_cheap_cpp.cpp)
    target_link_libraries(bench_cheap_cpp PRIVATE cheap_cpp)
    target_compile_definitions(bench_cheap_cpp PRIVATE _POSIX_C_SOURCE=199309L)

    # C++ examples
    set(CHEAP_EXAMPLES_CPP
        gp_regression
        toeplitz_solve
        optimal_transport
        lqr_mpc
        fractional_diff
        poisson_solver
        ns_dissipation
        online_filter
        wiener_denoise
        rmt_denoise
        poisson_solver_2d
        poisson_solver_3d
        wiener_denoise_2d
        ns_dissipation_2d
        ns_dissipation_3d
        gp_regression_2d
        toeplitz_solve_2d
    )

    foreach(example ${CHEAP_EXAMPLES_CPP})
        add_executable(${example}_cpp examples/cpp/${example}.cpp)
        target_link_libraries(${example}_cpp PRIVATE cheap_cpp)
        target_compile_definitions(${example}_cpp PRIVATE _POSIX_C_SOURCE=199309L)
    endforeach()
endif()
