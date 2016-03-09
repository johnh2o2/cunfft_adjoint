/*   cufft_utils.cu
 *   ==============
 *
 *   cunfft error parsing and printing
 *
 *   (c) John Hoffman 2016
 *
 *   This file is part of CUNA
 *
 *   CUNA is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CUNA is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CUNA.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <cufft.h>

char * cufftParseError(cufftResult_t r){
    char *message = (char *) malloc( 100 * sizeof(char));
    switch(r){
        case CUFFT_SUCCESS:
            sprintf(message, "The cuFFT operation was successful.");
            return message;
        case CUFFT_INVALID_PLAN:
            sprintf(message, "cuFFT was passed an invalid plan handle.");
            return message;
        case CUFFT_ALLOC_FAILED:
            sprintf(message, "cuFFT failed to allocate GPU or CPU memory.");
            return message;
        case CUFFT_INVALID_TYPE:
            sprintf(message, "CUFFT_INVALID_TYPE (no longer used)");
            return message;
        case CUFFT_INVALID_VALUE:
            sprintf(message, "User specified an invalid pointer or parameter");
            return message;
        case CUFFT_INTERNAL_ERROR:
            sprintf(message, "Driver or internal cuFFT library error.");
            return message;
        case CUFFT_EXEC_FAILED:
            sprintf(message, "Failed to execute an FFT on the GPU.");
            return message;
        case CUFFT_SETUP_FAILED:
            sprintf(message, "The cuFFT library failed to initialize.");
            return message;
        case CUFFT_INVALID_SIZE:
            sprintf(message, "User specified an invalid transform size.");
            return message;
        case CUFFT_UNALIGNED_DATA:
            sprintf(message, "CUFFT_UNALIGNED_DATA (no longer used).");
            return message;
        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            sprintf(message, "Missing parameters in call.");
            return message;
        case CUFFT_INVALID_DEVICE:
            sprintf(message, "Execution of a plan was on different GPU than plan creation. ");
            return message;
        case CUFFT_PARSE_ERROR:
            sprintf(message, "Internal plan database error.");
            return message;
        case CUFFT_NO_WORKSPACE:
            sprintf(message, "No workspace has been provided prior to plan execution.");
            return message;
        default:
            sprintf(message, "DONT UNDERSTAND THE CUFFT ERROR CODE!! %d", r);
            return message;
    }
}

void checkCufftError(cufftResult_t r){
    if (r == CUFFT_SUCCESS) return;

    fprintf(stderr, "cuFFT ERROR: %s\n", cufftParseError(r));
    exit(r);
}
