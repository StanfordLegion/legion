/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#ifndef LEGION_LOWLEVEL_LAYOUTS
#define LEGION_LOWLEVEL_LAYOUTS
#include "arrays.h"

using namespace LegionRuntime::Arrays;

namespace LegionRuntime {
  namespace Layouts {
    class XferOrder {
    public:
      enum Type {
        SRC_FIFO,
        DST_FIFO,
        ANY_ORDER
      };
    };

    enum DimKind {
      DIM_X,
      DIM_Y,
      DIM_Z
    };

    template <unsigned DIM>
    class SplitDimLinearization {
    protected:
      unsigned dim_sum;
      DimKind dim_kind[DIM * 2];
      size_t dim_size[DIM * 2];
    public:
      enum { IDIM = DIM, ODIM = 1};
      SplitDimLinearization() {}
      SplitDimLinearization(std::vector<DimKind> dim_kind_vec, std::vector<size_t> dim_size_vec)
      {
        dim_sum = dim_kind_vec.size();
        assert(dim_size_vec.size() == dim_sum);
        for (unsigned i = 0; i < dim_sum; i++) {
          dim_kind[i] = dim_kind_vec[i];
          dim_size[i] = dim_size_vec[i];
        }
      }

      Point<ODIM> image(const Point<IDIM> p) const
      {
        int index = 0, subtotal = 1;
        Point<IDIM> local_p = p;
        for (unsigned i = 0; i < dim_sum; i++) {
          switch (dim_kind[i]) {
            case DIM_X:
              index += (local_p.x[0] % dim_size[i]) * subtotal;
              subtotal *= dim_size[i];
              local_p.x[0] /= dim_size[i];
              break;
            case DIM_Y:
              index += (local_p.x[1] % dim_size[i]) * subtotal;
              subtotal *= dim_size[i];
              local_p.x[1] /= dim_size[i];
              break;
            case DIM_Z:
              index += (local_p.x[DIM-1] % dim_size[i]) * subtotal;
              subtotal *= dim_size[i];
              local_p.x[DIM-1] /= dim_size[i];
              break;
            default:
              assert(0);
          }
        }
        return make_point(index);
      }

      Rect<1> image_convex(const Rect<IDIM> r) const
      {
        return Rect<1>(image(r.lo), image(r.hi));
      }

      bool image_is_dense(const Rect<IDIM> r) const
      {
        assert(0);
        return false;
      }

      Rect<ODIM> image_dense_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect) const
      {
        assert(0);
        return Rect<ODIM> (Point<ODIM>::ZEROES(), Point<ODIM>::ZEROES());
      }

      Point<ODIM> image_linear_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect, Point<ODIM> strides[IDIM]) const
      {
        subrect = r;
        for (unsigned i = 0; i < IDIM; i++)
          strides[i] = Point<ODIM>::ZEROES();
        int subtotal = 1;
        for (unsigned i = 0; i < dim_sum; i++) {
          switch (dim_kind[i]) {
            case DIM_X:
              if (strides[0][0] == 0) {
                strides[0].x[0] = subtotal;
                subrect.hi.x[0] = r.lo.x[0] - r.lo.x[0] % dim_size[i] + dim_size[i] - 1;
              }
              subtotal *= dim_size[i];
              break;
            case DIM_Y:
              if (strides[1][0] == 0) {
                strides[1].x[0] = subtotal;
                subrect.hi.x[1] = r.lo.x[1] - r.lo.x[1] % dim_size[i] + dim_size[i] - 1;
              }
              subtotal *= dim_size[i];
              break;
            case DIM_Z:
              if (strides[2][0] == 0) {
                strides[2].x[0] = subtotal;
                subrect.hi.x[2] = r.lo.x[2] - r.lo.x[2] % dim_size[i] + dim_size[i] - 1;
              }
              subtotal *= dim_size[i];
              break;
            default:
              assert(0);
          }
        }
        return image(r.lo);
      }

      Rect<IDIM> preimage(const Point<ODIM> p) const
      {
        assert(ODIM == 1);
        int index = p.x[0];
        Point<IDIM> ret = Point<IDIM>::ZEROES(), dim_base = Point<IDIM>::ONES();
        for (unsigned i = 0; i < dim_sum; i++) {
          switch (dim_kind[i]) {
            case DIM_X:
              ret.x[0] += dim_base[0] * (index % dim_size[i]);
              dim_base.x[0] *= dim_size[i];
              index /= dim_size[i];
              break;
            case DIM_Y:
              ret.x[1] += dim_base[1] * (index % dim_size[i]);
              dim_base.x[1] *= dim_size[i];
              index /= dim_size[i];
              break;
            case DIM_Z:
              ret.x[DIM-1] += dim_base[DIM-1] * (index % dim_size[i]);
              dim_base.x[DIM-1] *= dim_size[i];
              index /= dim_size[i];
              break;
            default:
              assert(0);
          }
        }
        return Rect<IDIM> (ret, ret);
      }

      bool preimage_is_dense(const Point<ODIM> p) const
      {
        assert(0);
        return false;
      }
#ifdef CONTINUOUS_STEPS
      int continuous_steps(const Point<IDIM> p, int &direction) const
      {
        switch (dim_kind[0]) {
          case DIM_X:
            direction = 0;
            return dim_size[0] - p[0] % dim_size[0];
            break;
          case DIM_Y:
            direction = 1;
            return dim_size[0] - p[1] % dim_size[0];
            break;
          case DIM_Z:
            direction = 2;
            return dim_size[0] - p[2] % dim_size[0];
            break;
          default:
            assert(0);
        }
        return 0;
      }
#endif
    };

    struct OffsetsAndSize {
      off_t src_offset, dst_offset;
      int size;
    };

    template<unsigned DIM>
    class GenericLayoutIterator {
    public:
      GenericLayoutIterator(Rect<DIM> rect, Mapping<DIM, 1> *src_m, Mapping<DIM, 1> *dst_m, XferOrder::Type order)
      : orig_rect(rect), src_mapping(src_m), dst_mapping(dst_m), iter_order(order), cur_idx(0)
      {
        src_mapping->add_reference();
        dst_mapping->add_reference();
        rect_size = orig_rect.volume();
      }
      ~GenericLayoutIterator()
      {
        src_mapping->remove_reference();
        dst_mapping->remove_reference();
      }

      void reset()
      {
        cur_idx = 0;
      }

      bool any_left()
      {
        return cur_idx < rect_size;
      }

      int continuous_steps(int &src_idx, int &dst_idx)
      {
        Rect<DIM> r, src_subrect, dst_subrect;
        Point<1> src_strides[DIM], dst_strides[DIM];
        int subtotal = 1;
        switch(iter_order) {
          case XferOrder::SRC_FIFO:
            r = src_mapping->preimage(make_point(cur_idx));
            assert(r.volume() == 1);
            r.hi = orig_rect.hi;
            src_idx = src_mapping->image_linear_subrect(r, src_subrect, src_strides);
            dst_idx = dst_mapping->image_linear_subrect(r, dst_subrect, dst_strides);
            break;
          case XferOrder::DST_FIFO:
          case XferOrder::ANY_ORDER:
            r = dst_mapping->preimage(make_point(cur_idx));
            assert(r.volume() == 1);
            r.hi = orig_rect.hi;
            src_idx = src_mapping->image_linear_subrect(r, src_subrect, src_strides);
            dst_idx = dst_mapping->image_linear_subrect(r, dst_subrect, dst_strides);
            break;
          default:
            assert(0);
        }

        for (int i = 0; i < DIM; i++) {
          for (int j = 0; j < DIM; j++) {
            if (src_strides[j][0] == subtotal && dst_strides[j][0] == subtotal) {
              subtotal = subtotal * imin(src_subrect.dim_size(j), dst_subrect.dim_size(j));
            }
          }
        }

        return subtotal;
      }
      void move(int steps)
      {
        cur_idx += steps;
        assert(cur_idx <= rect_size);
      }
    private:
      Rect<DIM> orig_rect;
      Mapping<DIM, 1> *src_mapping, *dst_mapping;
      XferOrder::Type iter_order;
      size_t cur_idx, rect_size;
    };

  } // namespace Layout
} // namespace LegionRuntime
#endif
