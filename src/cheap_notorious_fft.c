/* Compile Notorious-FFT implementation once for the CHEAP CMake backend.
 * Consumers using CHEAP_USE_NOTORIOUS_FFT outside CMake must likewise
 * provide exactly one NOTORIOUS_FFT_IMPLEMENTATION translation unit.
 */
#define NOTORIOUS_FFT_IMPLEMENTATION
#include "notorious_fft.h"
