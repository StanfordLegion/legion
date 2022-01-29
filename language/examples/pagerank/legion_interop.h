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

#ifndef __LEGION_INTEROP_H__
#define __LEGION_INTEROP_H__

#ifdef __cplusplus
extern "C" {
#endif

enum {
  TID_F = 1,
  TID_F2 = 2,
};

enum FieldIDs {
  FID_DATA = 101,
};

void register_tasks();

#ifdef __cplusplus
}
#endif

#endif // __LEGION_INTEROP_H__

