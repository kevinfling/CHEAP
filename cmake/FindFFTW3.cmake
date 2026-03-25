# FindFFTW3.cmake — locate FFTW3 on systems without pkg-config
# (macOS Homebrew, custom install prefix, etc.)
#
# Imported target: FFTW3::fftw3
# Cache variables:
#   FFTW3_INCLUDE_DIR  — directory containing fftw3.h
#   FFTW3_LIBRARY      — path to libfftw3

include(FindPackageHandleStandardArgs)

# Search hints
set(_fftw3_hints
    ${FFTW3_ROOT}
    $ENV{FFTW3_ROOT}
    /usr/local
    /opt/homebrew          # macOS Apple-silicon Homebrew
    /opt/homebrew/opt/fftw # macOS explicit keg
    /usr/local/opt/fftw    # macOS Intel Homebrew
    /opt/local             # MacPorts
)

find_path(FFTW3_INCLUDE_DIR
    NAMES fftw3.h
    HINTS ${_fftw3_hints}
    PATH_SUFFIXES include
    DOC "Directory containing fftw3.h")

find_library(FFTW3_LIBRARY
    NAMES fftw3
    HINTS ${_fftw3_hints}
    PATH_SUFFIXES lib lib64
    DOC "Path to libfftw3")

find_package_handle_standard_args(FFTW3
    REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR)

if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3)
    add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
        IMPORTED_LOCATION             "${FFTW3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
endif()

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY)
