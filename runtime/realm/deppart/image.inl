/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include "realm/deppart/deppart_config.h"
#include "realm/deppart/image.h"
#include "realm/deppart/inst_helper.h"
#include "realm/deppart/preimage.h"
#include "realm/deppart/rectlist.h"
#include "realm/logging.h"

namespace Realm {

extern Logger log_part;
extern Logger log_uop_timing;

////////////////////////////////////////////////////////////////////////
//
// class TranslateImageMicroOp<N,T,N2,T2>

template <int N, typename T, int N2, typename T2>
TranslateImageMicroOp<N, T, N2, T2>::TranslateImageMicroOp(
    const IndexSpace<N, T> &_parent,
    const std::vector<IndexSpace<N2, T2>> &_sources,
    const TranslationTransform<N, T2> &_transform)
    : StructuredImageMicroOpBase<N, T, N2, T2>(_parent, _sources),
      transform(_transform) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N2; j++) {
      scale[i][j] = (i == j);
    }
  }
}

template <int N, typename T, int N2, typename T2>
TranslateImageMicroOp<N, T, N2, T2>::~TranslateImageMicroOp() {}

template <int N, typename T, int N2, typename T2>
inline void TranslateImageMicroOp<N, T, N2, T2>::populate(
    std::map<int, HybridRectangleList<N, T> *> &bitmasks) {
  std::vector<Rect<N, T>> parent_rects;
  if (this->parent_space.dense()) {
    parent_rects.push_back(this->parent_space.bounds);
  } else {
    for (IndexSpaceIterator<N, T> parent_it(this->parent_space);
         parent_it.valid; parent_it.step()) {
      parent_rects.push_back(parent_it.rect);
    }
  }

  assert(!parent_rects.empty());
  Rect<N, T> parent_bbox = parent_rects[0];
  for (size_t i = 1; i < parent_rects.size(); i++) {
    parent_bbox = parent_bbox.union_bbox(parent_rects[i]);
  }

  for (size_t i = 0; i < this->sources.size(); i++) {
    for (IndexSpaceIterator<N2, T2> it2(this->sources[i]); it2.valid;
         it2.step()) {
      Rect<N, T> source_bbox;
      source_bbox.lo = transform[scale * it2.rect.lo];
      source_bbox.hi = transform[scale * it2.rect.hi];

      if (parent_bbox.intersection(source_bbox).empty()) continue;

      for (const auto &parent_rect : parent_rects) {
        Rect<N, T> source_parent_isect = parent_rect.intersection(source_bbox);
        if (!source_parent_isect.empty()) {
          HybridRectangleList<N, T> **bmpp = 0;
          if (!bmpp) bmpp = &bitmasks[i];
          if (!*bmpp) *bmpp = new HybridRectangleList<N, T>;
          (*bmpp)->add_rect(source_parent_isect);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
//
// class AffineImageMicroOp<N,T,N2,T2>

template <int N, typename T, int N2, typename T2>
AffineImageMicroOp<N, T, N2, T2>::AffineImageMicroOp(
    const IndexSpace<N, T> &_parent,
    const std::vector<IndexSpace<N2, T2>> &_sources,
    const AffineTransform<N, N2, T2> &_transform)
    : StructuredImageMicroOpBase<N, T, N2, T2>(_parent, _sources),
      transform(_transform) {}

template <int N, typename T, int N2, typename T2>
AffineImageMicroOp<N, T, N2, T2>::~AffineImageMicroOp() {}

template <int N, typename T, int N2, typename T2>
inline void AffineImageMicroOp<N, T, N2, T2>::populate(
    std::map<int, HybridRectangleList<N, T> *> &bitmasks) {
  std::vector<Rect<N, T>> parent_rects;
  if (this->parent_space.dense()) {
    parent_rects.push_back(this->parent_space.bounds);
  } else {
    for (IndexSpaceIterator<N, T> parent_it(this->parent_space);
         parent_it.valid; parent_it.step()) {
      parent_rects.push_back(parent_it.rect);
    }
  }

  assert(!parent_rects.empty());
  Rect<N, T> parent_bbox = parent_rects[0];
  for (size_t i = 1; i < parent_rects.size(); i++) {
    parent_bbox = parent_bbox.union_bbox(parent_rects[i]);
  }

  for (size_t i = 0; i < this->sources.size(); i++) {
    for (IndexSpaceIterator<N2, T2> it2(this->sources[i]); it2.valid;
         it2.step()) {
      for (PointInRectIterator<N2, T2> point(it2.rect); point.valid;
           point.step()) {
        Point<N, T> source_point = transform[point.p];

        if (!parent_bbox.contains(source_point)) continue;

        for (const auto &parent_rect : parent_rects) {
          if (parent_rect.contains(source_point)) {
            HybridRectangleList<N, T> **bmpp = 0;
            if (!bmpp) bmpp = &bitmasks[i];
            if (!*bmpp) *bmpp = new HybridRectangleList<N, T>;
            (*bmpp)->add_point(source_point);
          }
        }
      }
    }
  }
}

}  // namespace Realm
