/* Copyright 2022 Stanford University
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

#include "cpu_kernels.h"

void stencil(DTYPE* RESTRICT inputPtr,
             DTYPE* RESTRICT outputPtr,
             DTYPE* RESTRICT weightPtr,
             coord_t haloX, coord_t startX, coord_t endX,
             coord_t startY, coord_t endY)
{
#define IN(i, j)     inputPtr[(j) * haloX + i]
#define OUT(i, j)    outputPtr[(j) * haloX + i]
#define WEIGHT(i, j) weightPtr[(j + RADIUS) * (2 * RADIUS + 1) + (i + RADIUS)]
  for (coord_t j = startY; j < endY; ++j)
    for (coord_t i = startX; i < endX; ++i)
      {
        for (coord_t jj = -RADIUS; jj <= RADIUS; jj++)
          OUT(i, j) += WEIGHT(0, jj) * IN(i, j + jj);
        for (coord_t ii = -RADIUS; ii < 0; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
        for (coord_t ii = 1; ii <= RADIUS; ii++)
          OUT(i, j) += WEIGHT(ii, 0) * IN(i + ii, j);
      }
#undef IN
#undef OUT
#undef WEIGHT
}

void increment(DTYPE* RESTRICT inputPtr,
               coord_t haloX, coord_t startX, coord_t endX,
               coord_t startY, coord_t endY)
{
#define IN(i, j)     inputPtr[(j) * haloX + i]
  for (coord_t j = startY; j < endY; ++j)
    for (coord_t i = startX; i < endX; ++i)
      {
        IN(i, j) += 1;
      }
#undef IN
#undef OUT
#undef WEIGHT
}

void copy2D(DTYPE* RESTRICT inputPtr,
            DTYPE* RESTRICT outputPtr,
            coord_t haloX, coord_t startX, coord_t endX,
            coord_t startY, coord_t endY,
            coord_t outputHaloX, coord_t outputStartX,
            coord_t outputStartY)
{
#define IN(i, j)     inputPtr[(j) * haloX + i]
#define OUT(i, j)    outputPtr[(j - (outputStartY - startY)) * outputHaloX + (i - (outputStartX - startX))]
  for (coord_t j = startY; j < endY; ++j)
    for (coord_t i = startX; i < endX; ++i)
      OUT(i, j) = IN(i, j);
#undef IN
#undef OUT
}
