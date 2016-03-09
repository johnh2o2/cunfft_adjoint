/*   cufft_utils.h
 *   =============
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
#ifndef CUFFT_UTILS_H
#define CUFFT_UTILS_H

#include <cufft.h>

__host__ char * cufftParseError(cufftResult_t r);

__host__ void checkCufftError(cufftResult_t r);
#endif
