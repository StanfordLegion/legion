
/* Copyright 2024 NVIDIA Corporation
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

#ifndef UCP_UTILS_H
#define UCP_UTILS_H

#define CHKERR_ACTION(_cond, _msg, _log, _action) \
    do { \
        if (_cond) { \
            _log.error() << _msg; \
            _action; \
        } \
    } while (0)


#define CHKERR_JUMP(_cond, _msg, _log, _label) \
    CHKERR_ACTION(_cond, _msg, _log, goto _label)

#define IS_POW2_OR_ZERO(_n) \
    !((_n) & ((_n) - 1))

#define IS_POW2(_n) \
    (((_n) > 0) && IS_POW2_OR_ZERO(_n))

#define PADDING_TO_ALIGN(_n, _alignment) \
    ( ((_alignment) - (_n) % (_alignment)) % (_alignment) )

#define ALIGN_DOWN_POW2(_n, _alignment) \
    ( (_n) & ~((_alignment) - 1) )

#define ALIGN_UP_POW2(_n, _alignment) \
    ALIGN_DOWN_POW2((_n) + (_alignment) - 1, _alignment)

#endif
