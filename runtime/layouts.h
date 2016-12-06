/* Copyright 2016 Stanford University, NVIDIA Corporation
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

    static inline coord_t BLOCK_UP(coord_t a, coord_t b) { return a - a % b + b - 1; }
    static inline int min(int a, int b) { return (a < b) ? a : b; }
    static inline coord_t min(coord_t a, coord_t b) { return (a < b) ? a : b; }

    template <unsigned DIM>
    class SplitDimLinearization {
    public:
      enum { IDIM = DIM, ODIM = 1};
    protected:
      unsigned dim_sum;
      DimKind dim_kind[DIM * 2];
      size_t dim_size[DIM * 2];
      Point<IDIM> lo_in;
      Point<ODIM> lo_out;
    public:
      SplitDimLinearization() {}
      SplitDimLinearization(Point<IDIM> lo_input, Point<ODIM> lo_output, std::vector<DimKind> dim_kind_vec, std::vector<size_t> dim_size_vec)
      {
        lo_in = lo_input;
        lo_out = lo_output;
        dim_sum = dim_kind_vec.size();
        assert(dim_size_vec.size() == dim_sum);
        for (unsigned i = 0; i < dim_sum; i++) {
          dim_kind[i] = dim_kind_vec[i];
          dim_size[i] = dim_size_vec[i];
        }
      }

      Point<ODIM> image(const Point<IDIM> p) const
      {
        coord_t index = 0, subtotal = 1;
        Point<IDIM> local_p = p - lo_in;
        unsigned x_idx = 0, y_idx = 1, z_idx = 2;
        for (unsigned i = 0; i < dim_sum; i++) {
          switch (dim_kind[i]) {
            case DIM_X:
              index += (local_p.x[x_idx] % dim_size[i]) * subtotal;
              subtotal *= dim_size[i];
              local_p.x[x_idx] /= dim_size[i];
              break;
            case DIM_Y:
              index += (local_p.x[y_idx] % dim_size[i]) * subtotal;
              subtotal *= dim_size[i];
              local_p.x[y_idx] /= dim_size[i];
              break;
            case DIM_Z:
              index += (local_p.x[z_idx] % dim_size[i]) * subtotal;
              subtotal *= dim_size[i];
              local_p.x[z_idx] /= dim_size[i];
              break;
            default:
              assert(0);
          }
        }
        return make_point(index) + lo_out;
      }

      Rect<1> image_convex(const Rect<IDIM> r) const
      {
        return Rect<1>(image(r.lo), image(r.hi));
      }

      bool image_is_dense(const Rect<IDIM> r) const
      {
        Rect<1> convex = image_convex(r);
        coord_t prod = 1;
        for (unsigned i = 0; i < IDIM; i++) {
          prod *= 1 + (r.hi[i] - r.lo[i]);
        }
        return (convex.hi[0] - convex.lo[0] + 1) == prod;
      }

      Rect<ODIM> image_dense_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect) const
      {
        Rect<IDIM> local_r(r.lo - lo_in, r.hi - lo_in);
        Rect<IDIM> local_s(local_r.lo, local_r.lo);
        // TODO: for now each dimension can fold at most once
        coord_t count = 1, subtotal = 1;
        std::vector<bool> folded(3, false);
        for (unsigned i = 0; i < dim_sum; i++) {
          if (count != subtotal) break;
          unsigned idx;
          switch (dim_kind[i]) {
            case DIM_X:
              idx = 0; break;
            case DIM_Y:
              idx = 1; break;
            case DIM_Z:
              idx = 2; break;
            default:
              assert(0);
          }
          if (folded[idx]) break;
          local_s.hi.x[idx] = min(BLOCK_UP(local_r.lo.x[idx], dim_size[i]),
                                  local_r.hi.x[idx]);
          count *= (local_s.hi.x[idx] - local_s.lo.x[idx] + 1);
          folded[idx] = true;
          subtotal *= dim_size[i];
        }
        subrect = Rect<IDIM>(local_s.lo + lo_in, local_s.hi + lo_in);
        assert(image_is_dense(subrect));
        return image_convex(subrect);
      }

      Point<ODIM> image_linear_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect, Point<ODIM> strides[IDIM]) const
      {
        Rect<IDIM> local_r(r.lo - lo_in, r.hi - lo_in);
        Rect<IDIM> local_subrect(local_r.lo, local_r.lo);
        for (unsigned i = 0; i < IDIM; i++)
          strides[i] = Point<ODIM>::ZEROES();
        coord_t subtotal = 1;
        unsigned x_idx = 0, y_idx = 1, z_idx = 2;
        for (unsigned i = 0; i < dim_sum; i++) {
          switch (dim_kind[i]) {
            case DIM_X:
              if (strides[x_idx][0] == 0) {
                strides[x_idx].x[0] = subtotal;
                local_subrect.hi.x[x_idx] = min(local_r.lo.x[x_idx] - local_r.lo.x[x_idx] % dim_size[i] + dim_size[i] - 1, local_r.hi.x[x_idx]);
              }
              subtotal *= dim_size[i];
              break;
            case DIM_Y:
              if (strides[y_idx][0] == 0) {
                strides[y_idx].x[0] = subtotal;
                local_subrect.hi.x[y_idx] = min(local_r.lo.x[y_idx] - local_r.lo.x[y_idx] % dim_size[i] + dim_size[i] - 1, local_r.hi.x[y_idx]);
              }
              subtotal *= dim_size[i];
              break;
            case DIM_Z:
              if (strides[z_idx][0] == 0) {
                strides[z_idx].x[0] = subtotal;
                local_subrect.hi.x[z_idx] = min(local_r.lo.x[z_idx] - local_r.lo.x[z_idx] % dim_size[i] + dim_size[i] - 1, local_r.hi.x[z_idx]);
              }
              subtotal *= dim_size[i];
              break;
            default:
              assert(0);
          }
        }
        subrect = Rect<IDIM>(local_subrect.lo + lo_in, local_subrect.hi + lo_in);
        return image(r.lo);
      }

      Rect<IDIM> preimage(const Point<ODIM> p) const
      {
        assert(ODIM == 1);
        coord_t index = p.x[0] - lo_out.x[0];
        Point<IDIM> ret = Point<IDIM>::ZEROES(), dim_base = Point<IDIM>::ONES();
        unsigned x_idx = 0, y_idx = 1, z_idx = 2;
        for (unsigned i = 0; i < dim_sum; i++) {
          switch (dim_kind[i]) {
            case DIM_X:
              ret.x[x_idx] += dim_base[x_idx] * (index % dim_size[i]);
              dim_base.x[x_idx] *= dim_size[i];
              index /= dim_size[i];
              break;
            case DIM_Y:
              ret.x[y_idx] += dim_base[y_idx] * (index % dim_size[i]);
              dim_base.x[y_idx] *= dim_size[i];
              index /= dim_size[i];
              break;
            case DIM_Z:
              ret.x[z_idx] += dim_base[z_idx] * (index % dim_size[i]);
              dim_base.x[z_idx] *= dim_size[i];
              index /= dim_size[i];
              break;
            default:
              assert(0);
          }
        }
        return Rect<IDIM>(ret + lo_in, ret + lo_in);
      }

      bool preimage_is_dense(const Point<ODIM> p) const
      {
        assert(0);
        return false;
      }
#ifdef CONTINUOUS_STEPS
      coord_t continuous_steps(const Point<IDIM> p, int &direction) const
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
      coord_t size;
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
        switch(iter_order) {
          case XferOrder::SRC_FIFO:
            idx_offset = src_mapping->image(rect.lo);
            break;
          case XferOrder::DST_FIFO:
          case XferOrder::ANY_ORDER:
            idx_offset = dst_mapping->image(rect.lo);
            break;
          default:
            assert(0);
        }
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

      coord_t continuous_steps(coord_t &src_idx, coord_t &dst_idx)
      {
        Rect<DIM> r, src_subrect, dst_subrect;
        Point<1> src_strides[DIM], dst_strides[DIM];
        coord_t subtotal = 1;
        coord_t idx = cur_idx;
        for (unsigned i = 0; i < DIM; i++) {
          r.lo.x[i] = idx % orig_rect.dim_size(i) + orig_rect.lo.x[i];
          idx = idx / orig_rect.dim_size(i);
        }
        r.hi = orig_rect.hi;
        switch(iter_order) {
          case XferOrder::SRC_FIFO:
            //r = src_mapping->preimage(make_point(cur_idx + idx_offset));
            //assert(r.volume() == 1);
            //r.hi = orig_rect.hi;
            src_idx = src_mapping->image_linear_subrect(r, src_subrect, src_strides);
            dst_idx = dst_mapping->image_linear_subrect(r, dst_subrect, dst_strides);
            break;
          case XferOrder::DST_FIFO:
          case XferOrder::ANY_ORDER:
            //r = dst_mapping->preimage(make_point(cur_idx + idx_offset));
            //assert(r.volume() == 1);
            //r.hi = orig_rect.hi;
            dst_idx = dst_mapping->image_linear_subrect(r, dst_subrect, dst_strides);
            src_idx = src_mapping->image_linear_subrect(r, src_subrect, src_strides);
            break;
          default:
            assert(0);
        }

        for (unsigned i = 0; i < DIM; i++) {
          for (unsigned j = 0; j < DIM; j++) {
            if (src_strides[j][0] == subtotal && dst_strides[j][0] == subtotal) {
              subtotal = subtotal * imin(src_subrect.dim_size(j), dst_subrect.dim_size(j));
            }
          }
        }

        return subtotal;
      }

      // TODO: currently only support FortranArrayLinearization
      coord_t continuous_steps(coord_t &src_idx, coord_t &dst_idx,
                               coord_t &src_stride, coord_t &dst_stride,
                               size_t &items_per_line, size_t &nlines)
      {
        Rect<DIM> r, src_subrect, dst_subrect;
        Point<1> src_strides[DIM], dst_strides[DIM];
        coord_t subtotal = 1;
        coord_t idx = cur_idx;
        for (unsigned i = 0; i < DIM; i++) {
          r.lo.x[i] = idx % orig_rect.dim_size(i) + orig_rect.lo.x[i];
          idx = idx / orig_rect.dim_size(i);
        }
        r.hi = orig_rect.hi;
        src_idx = src_mapping->image_linear_subrect(r, src_subrect, src_strides);
        dst_idx = dst_mapping->image_linear_subrect(r, dst_subrect, dst_strides);

        bool mergeable = true;
        for (unsigned j = 0; j < DIM; j++) {
          if (src_strides[j][0] == subtotal && dst_strides[j][0] == subtotal) {
            if (src_subrect.dim_size(j) != orig_rect.dim_size(j))
              mergeable = false;
            subtotal = subtotal * imin(src_subrect.dim_size(j), dst_subrect.dim_size(j));
          }
        }

        if (iter_order == XferOrder::SRC_FIFO) {
          for (unsigned i = 0; i < DIM; i++)
            if (src_strides[i][0] == subtotal && mergeable) {
              src_stride = src_strides[i][0];
              dst_stride = dst_strides[i][0];
              items_per_line = subtotal;
              nlines = imin(src_subrect.dim_size(i), dst_subrect.dim_size(i));
              return items_per_line * nlines;
            }
        } else if (iter_order == XferOrder::DST_FIFO) {
          for (unsigned i = 0; i < DIM; i++)
            if (dst_strides[i][0] == subtotal && mergeable) {
              src_stride = src_strides[i][0];
              dst_stride = dst_strides[i][0];
              items_per_line = subtotal;
              nlines = imin(src_subrect.dim_size(i), dst_subrect.dim_size(i));
              return items_per_line * nlines;
            }
        } else if (iter_order == XferOrder::ANY_ORDER) {
          for (unsigned i = 0; i < DIM; i++)
            if (dst_strides[i][0] >= subtotal && mergeable) {
              src_stride = src_strides[i][0];
              dst_stride = dst_strides[i][0];
              items_per_line = subtotal;
              nlines = imin(src_subrect.dim_size(i), dst_subrect.dim_size(i));
              return items_per_line * nlines;
            }
        }
        src_stride = subtotal;
        dst_stride = subtotal;
        items_per_line = subtotal;
        nlines = 1;
        return subtotal;
      }

      // TODO: currently only support FortranArrayLinearization
      coord_t continuous_steps(coord_t &src_idx, coord_t &dst_idx,
                               coord_t &src_stride, coord_t &dst_stride,
                               off_t &src_height, off_t &dst_height,
                               size_t &items_per_line, size_t &height, size_t &depth)
      {
        Rect<DIM> r, src_subrect, dst_subrect;
        Point<1> src_strides[DIM], dst_strides[DIM];
        coord_t subtotal = 1;
        coord_t idx = cur_idx;
        for (unsigned i = 0; i < DIM; i++) {
          r.lo.x[i] = idx % orig_rect.dim_size(i) + orig_rect.lo.x[i];
          idx = idx / orig_rect.dim_size(i);
        }
        r.hi = orig_rect.hi;
        src_idx = src_mapping->image_linear_subrect(r, src_subrect, src_strides);
        dst_idx = dst_mapping->image_linear_subrect(r, dst_subrect, dst_strides);

        bool mergeable = true;
        for (unsigned j = 0; j < DIM; j++) {
          if (src_strides[j][0] == subtotal && dst_strides[j][0] == subtotal) {
            if (src_subrect.dim_size(j) != orig_rect.dim_size(j))
              mergeable = false;
            subtotal = subtotal * imin(src_subrect.dim_size(j), dst_subrect.dim_size(j));
          }
        }

        // see if we can use 3D/2D
        assert(DIM == 3);
        if (iter_order == XferOrder::SRC_FIFO) {
          if (src_strides[1][0] == subtotal && mergeable
          && src_strides[2][0] == subtotal * src_subrect.dim_size(1)){
//          &&(dst_strides[2][0] == subtotal * dst_strides[1][0])) {
            src_stride = src_strides[1][0];
            dst_stride = dst_strides[1][0];
            src_height = src_strides[2][0] / src_strides[1][0];
            dst_height = dst_strides[2][0] / dst_strides[1][0];
            items_per_line = subtotal;
            height = src_subrect.dim_size(1);
            depth = src_subrect.dim_size(2);
            return items_per_line * height * depth;
          }
          for (unsigned i = 0; i < DIM; i++)
            if (src_strides[i][0] == subtotal && mergeable) {
              src_stride = src_strides[i][0];
              dst_stride = dst_strides[i][0];
              items_per_line = subtotal;
              height = imin(src_subrect.dim_size(i), dst_subrect.dim_size(i));
              src_height = dst_height = height;
              depth = 1;
              return items_per_line * height;
            }
        } else if (iter_order == XferOrder::DST_FIFO) {
          if (dst_strides[1][0] == subtotal && mergeable
          && dst_strides[2][0] == subtotal * dst_subrect.dim_size(1)){
//          &&(dst_strides[2][0] == subtotal * dst_strides[1][0])) {
            src_stride = src_strides[1][0];
            dst_stride = dst_strides[1][0];
            src_height = src_strides[2][0] / src_strides[1][0];
            dst_height = dst_strides[2][0] / dst_strides[1][0];
            items_per_line = subtotal;
            height = dst_subrect.dim_size(1);
            depth = dst_subrect.dim_size(2);
            return items_per_line * height * depth;
          }
          for (unsigned i = 0; i < DIM; i++)
            if (dst_strides[i][0] == subtotal && mergeable) {
              src_stride = src_strides[i][0];
              dst_stride = dst_strides[i][0];
              items_per_line = subtotal;
              height = imin(src_subrect.dim_size(i), dst_subrect.dim_size(i));
              src_height = dst_height = height;
              depth = 1;
              return items_per_line * height;
            }
        } else if (iter_order == XferOrder::ANY_ORDER) {
          if (dst_strides[1][0] >= subtotal && mergeable) {
            src_stride = src_strides[1][0];
            dst_stride = dst_strides[1][0];
            src_height = src_strides[2][0] / src_strides[1][0];
            dst_height = dst_strides[2][0] / dst_strides[1][0];
            items_per_line = subtotal;
            height = dst_subrect.dim_size(1);
            depth = dst_subrect.dim_size(2);
            return items_per_line * height * depth;
          }
          for (unsigned i = 0; i < DIM; i++)
            if (dst_strides[i][0] >= subtotal && mergeable) {
              src_stride = src_strides[i][0];
              dst_stride = dst_strides[i][0];
              items_per_line = subtotal;
              height = imin(src_subrect.dim_size(i), dst_subrect.dim_size(i));
              src_height = dst_height = height;
              depth = 1;
              return items_per_line * height;
            }
        }

        src_stride = subtotal;
        dst_stride = subtotal;
        src_height = 1;
        dst_height = 1;
        items_per_line = subtotal;
        height = 1;
        depth = 1;
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
      coord_t idx_offset;
    };

#ifdef USE_HDF
   template<unsigned DIM>
   class HDFLayoutIterator {
   public:
     HDFLayoutIterator(Rect<DIM> rect, Mapping<DIM, 1> *m, size_t bs)
     : orig_rect(rect), mapping(m), cur_idx(0), rect_size(rect.volume()), block_size(bs)
     {
       mapping->add_reference();
       Rect<DIM> local_rect;
       cur_idx = mapping->image_linear_subrect(orig_rect, local_rect, strides);
       assert(cur_idx == 0);
       sub_rect = Rect<DIM>(local_rect.lo, local_rect.lo);
       size_t subtotal = 1;
       size_t left_over = block_size - cur_idx % block_size;
       for (int j = 0; j < DIM; j++) {
         if (strides[j][0] == subtotal && subtotal * local_rect.dim_size(j) <= left_over) {
           sub_rect.hi.x[j] = local_rect.hi[j];
           subtotal *= local_rect.dim_size(j);
         }
         else {
           if (strides[j][0] == subtotal) {
             size_t partial = left_over / subtotal;
             sub_rect.hi.x[j] += partial - 1;
             subtotal *= partial;
           }
           break;
         }
       }
     }

     ~HDFLayoutIterator()
     {
       mapping->remove_reference();
     }

     bool any_left()
     {
       return cur_idx < rect_size;
     }

     bool step()
     {
       cur_idx += sub_rect.volume();
       size_t left_over = block_size - cur_idx % block_size;
       if (!any_left())
         return false;
       Rect<DIM> r = mapping->preimage(make_point(cur_idx)), temp_rect;
       assert(r.volume() == 1);
       r.hi = orig_rect.hi;
       mapping->image_linear_subrect(r, temp_rect, strides);
       sub_rect = Rect<DIM>(temp_rect.lo, temp_rect.lo);
       coord_t subtotal = 1;
       for (int j = 0; j < DIM; j++) {
         if (strides[j][0] == subtotal && subtotal * temp_rect.dim_size(j) <= left_over) {
           sub_rect.hi.x[j] = temp_rect.hi[j];
           subtotal *= temp_rect.dim_size(j);
         }
         else
           break;
       }
       return true;
     }
   public:
     Rect<DIM> orig_rect, sub_rect;
     Point<1> strides[DIM];
     Mapping<DIM, 1> *mapping;
     size_t cur_idx, rect_size, block_size;
   };
#endif

  } // namespace Layout
} // namespace LegionRuntime
#endif
