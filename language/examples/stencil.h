/* Copyright 2018 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __STENCIL_H__
#define __STENCIL_H__

#ifndef DTYPE
#error DTYPE must be defined
#endif

#ifndef RESTRICT
#error RESTRICT must be defined
#endif

#ifndef RADIUS
#error RADIUS must be defined
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef long long int coord_t;

void stencil(DTYPE* RESTRICT inputPtr,
             DTYPE* RESTRICT outputPtr,
             coord_t haloX, coord_t startX, coord_t endX,
             coord_t startY, coord_t endY);

void increment(DTYPE* RESTRICT inputPtr,
               coord_t haloX, coord_t startX, coord_t endX,
               coord_t startY, coord_t endY);

#ifdef __cplusplus
}
#endif

#endif // __STENCIL_H__
