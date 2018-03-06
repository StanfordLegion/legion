/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// stuff to try making array accesses more sane

#ifndef _ARRAYS_H
#define _ARRAYS_H

#include <string.h>
#include <vector>
#include <map>
#include <ostream>

#include <assert.h>

#include <stdint.h>

#include "legion/legion_config.h"

#ifdef __CUDACC__
#define CUDAPREFIX __host__ __device__
#else
#define CUDAPREFIX
#endif

// why must I define this every time I need it?
CUDAPREFIX static inline ::legion_coord_t imin(::legion_coord_t a, ::legion_coord_t b) { return (a < b) ? a : b; }
CUDAPREFIX static inline ::legion_coord_t imax(::legion_coord_t a, ::legion_coord_t b) { return (a > b) ? a : b; }

namespace LegionRuntime {
  namespace Arrays {
    typedef ::legion_coord_t coord_t;

    template <unsigned DIM>
    class Point {
    public:
      CUDAPREFIX Point(void) {}
      CUDAPREFIX Point(const coord_t *vals) { for(unsigned i = 0; i < DIM; i++) x[i] = vals[i]; }
      CUDAPREFIX Point(const int *vals) { for(unsigned i = 0; i < DIM; i++) x[i] = vals[i]; }
      CUDAPREFIX Point(const Point<DIM>& other) { for(unsigned i = 0; i < DIM; i++) x[i] = other.x[i]; }

      CUDAPREFIX Point& operator=(const Point<DIM>& other) 
      { 
	for(unsigned i = 0; i < DIM; i++) x[i] = other.x[i];
	return *this;
      }

      CUDAPREFIX void to_array(coord_t *vals) const { for(unsigned i = 0; i < DIM; i++) vals[i] = x[i]; }

      CUDAPREFIX bool operator==(const Point<DIM>& other) const
      { 
	for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] != other.x[i]) return false; 
	return true; 
      }

      CUDAPREFIX bool operator!=(const Point<DIM>& other) const
      { 
	for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] != other.x[i]) return true; 
	return false; 
      }

      CUDAPREFIX bool operator<=(const Point<DIM>& other) const
      { 
	for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] > other.x[i]) return false; 
	return true; 
      }

      struct STLComparator {
	bool operator()(const Point<DIM>& a, const Point<DIM>& b) const
	{
	  for(unsigned i = 0; i < DIM; i++)  {
	    coord_t d = a.x[i] - b.x[i];
	    if (d < 0) return true;
	    if (d > 0) return false;
	  }
	  return false;
	}
      };

      CUDAPREFIX coord_t operator[](unsigned idx) const { return x[idx]; }
  
      CUDAPREFIX static Point<DIM> ZEROES(void)
      {
        Point<DIM> z; for(unsigned i = 0; i < DIM; i++) z.x[i] = 0; return z;
      }

      CUDAPREFIX static Point<DIM> ONES(void)
      {
        Point<DIM> o; for(unsigned i = 0; i < DIM; i++) o.x[i] = 1; return o;
      }

      CUDAPREFIX Point<DIM> operator+(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
          res.x[i] = x[i] + other.x[i];
	return res;
      }

      CUDAPREFIX Point<DIM> operator-(void) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
          res.x[i] = -x[i];
        return res;
      }
  
      CUDAPREFIX Point<DIM> operator-(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] - other.x[i];
	return res;
      }
  
      // element-wise multiplication and division
      CUDAPREFIX Point<DIM> operator*(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] * other.x[i];
	return res;
      }
  
      CUDAPREFIX Point<DIM> operator/(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] / other.x[i];
	return res;
      }

      CUDAPREFIX Point<DIM>& operator+=(const Point<DIM> &other)
      {
        for (unsigned i = 0; i < DIM; i++)
          x[i] += other.x[i];
        return *this;
      }

      CUDAPREFIX Point<DIM>& operator-=(const Point<DIM> &other)
      {
        for (unsigned i = 0; i < DIM; i++)
          x[i] -= other.x[i];
        return *this;
      }

      CUDAPREFIX Point<DIM>& operator*=(const Point<DIM> &other)
      {
        for (unsigned i = 0; i < DIM; i++)
          x[i] *= other.x[i];
        return *this;
      }

      CUDAPREFIX Point<DIM>& operator/=(const Point<DIM> &other)
      {
        for (unsigned i = 0; i < DIM; i++)
          x[i] /= other.x[i];
        return *this;
      }

      CUDAPREFIX static Point<DIM> sum(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = a.x[i] + b.x[i];
	return res;
      }

      CUDAPREFIX static Point<DIM> min(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = imin(a.x[i], b.x[i]);
	return res;
      }

      CUDAPREFIX static Point<DIM> max(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = imax(a.x[i], b.x[i]);
	return res;
      }

      CUDAPREFIX static coord_t dot(const Point<DIM> a, const Point<DIM> b)
      {
	coord_t v = 0;
        for(unsigned i = 0; i < DIM; i++)
	  v += a.x[i] * b.x[i];
	return v;
      }
  
      CUDAPREFIX coord_t dot(const Point<DIM> other) const
      {
        coord_t v = 0;
        for(unsigned i = 0; i < DIM; i++) v += x[i] * other.x[i];
        return v;
      }
  
    public:
      coord_t x[DIM];
    };
  
    template <>
    class Point<1> {
    public:
      enum { DIM = 1 };
      CUDAPREFIX Point(void) {}
      CUDAPREFIX Point(coord_t val) { x[0] = val; }
      CUDAPREFIX Point(int32_t val) { x[0] = val; }
      CUDAPREFIX Point(uint32_t val) { x[0] = val; }
      CUDAPREFIX Point(uint64_t val) { x[0] = val; }
#ifdef __MACH__
      // on Darwin, size_t is neither uint32_t nor uint64_t...
      CUDAPREFIX Point(size_t val) { x[0] = val; }
#endif
      CUDAPREFIX Point(const coord_t *vals) { for(unsigned i = 0; i < DIM; i++) x[i] = vals[i]; }
      CUDAPREFIX Point(const Point<1>& other) { for(unsigned i = 0; i < DIM; i++) x[i] = other.x[i]; }

      CUDAPREFIX Point& operator=(const Point<1>& other) 
      { 
	for(unsigned i = 0; i < 1; i++) x[i] = other.x[i];
	return *this;
      }

      CUDAPREFIX void to_array(coord_t *vals) const { for(unsigned i = 0; i < DIM; i++) vals[i] = x[i]; }
  
      CUDAPREFIX coord_t operator[](unsigned idx) const { return x[0]; }
      CUDAPREFIX operator int(void) const { return x[0]; }
      
      CUDAPREFIX bool operator==(const Point<DIM> &other) const
      {
        for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] != other.x[i]) return false; 
	return true;
      }

      CUDAPREFIX bool operator!=(const Point<DIM> &other) const
      {
        for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] != other.x[i]) return true; 
	return false;
      }

      CUDAPREFIX bool operator<=(const Point<DIM> &other) const
      {
        for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] > other.x[i]) return false; 
	return true;
      }
  
      CUDAPREFIX static Point<DIM> ZEROES(void)
      {
        Point<DIM> z; for(unsigned i = 0; i < DIM; i++) z.x[i] = 0; return z;
      }

      CUDAPREFIX static Point<DIM> ONES(void)
      {
        Point<DIM> o; for(unsigned i = 0; i < DIM; i++) o.x[i] = 1; return o;
      }

      CUDAPREFIX static Point<DIM> sum(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = a.x[i] + b.x[i];
	return res;
      }

      CUDAPREFIX static Point<DIM> min(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = imin(a.x[i], b.x[i]);
	return res;
      }

      CUDAPREFIX static Point<DIM> max(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = imax(a.x[i], b.x[i]);
	return res;
      }

      CUDAPREFIX static coord_t dot(const Point<DIM> a, const Point<DIM> b)
      {
	coord_t v = 0;
        for(unsigned i = 0; i < DIM; i++)
	  v += a.x[i] * b.x[i];
	return v;
      }

      CUDAPREFIX Point<DIM> operator+(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
          res.x[i] = x[i] + other.x[i];
	return res;
      }

      CUDAPREFIX Point<DIM> operator-(void) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
          res.x[i] = -x[i];
        return res;
      }

      CUDAPREFIX Point<DIM> operator-(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] - other.x[i];
	return res;
      }
  
      // element-wise multiplication and division
      CUDAPREFIX Point<DIM> operator*(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] * other.x[i];
	return res;
      }
  
      CUDAPREFIX Point<DIM> operator/(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] / other.x[i];
	return res;
      }

      CUDAPREFIX Point<DIM>& operator+=(const Point<DIM> &other)
      {
        for (unsigned i = 0; i < DIM; i++)
          x[i] += other.x[i];
        return *this;
      }

      CUDAPREFIX Point<DIM>& operator-=(const Point<DIM> &other)
      {
        for (unsigned i = 0; i < DIM; i++)
          x[i] -= other.x[i];
        return *this;
      }

      CUDAPREFIX Point<DIM>& operator*=(const Point<DIM> &other)
      {
        for (unsigned i = 0; i < DIM; i++)
          x[i] *= other.x[i];
        return *this;
      }

      CUDAPREFIX Point<DIM>& operator/=(const Point<DIM> &other)
      {
        for (unsigned i = 0; i < DIM; i++)
          x[i] /= other.x[i];
        return *this;
      }


      CUDAPREFIX coord_t dot(const Point<DIM> other) const { return dot(*this, other); }
  
    public:
      coord_t x[1];
    };

    CUDAPREFIX inline Point<1> make_point(coord_t x)
    {
      Point<1> p;
      p.x[0] = x;
      return p;
    }

    CUDAPREFIX inline Point<2> make_point(coord_t x, coord_t y)
    {
      Point<2> p;
      p.x[0] = x;
      p.x[1] = y;
      return p;
    }

    CUDAPREFIX inline Point<3> make_point(coord_t x, coord_t y, coord_t z)
    {
      Point<3> p;
      p.x[0] = x;
      p.x[1] = y;
      p.x[2] = z;
      return p;
    }

    template <unsigned DIM>
    inline std::ostream& operator<<(std::ostream& os, const Point<DIM>& p)
    {
      os << '(' << p[0];
      for(unsigned i = 1; i < DIM; i++)
	os << ',' << p[i];
      os << ')';
      return os;
    }

    template <unsigned DIM>
    class Rect {
    public:
      CUDAPREFIX Rect(void) {}
      CUDAPREFIX explicit Rect(const coord_t *vals) : lo(vals), hi(vals + DIM) {}
      CUDAPREFIX Rect(const Point<DIM> _lo, const Point<DIM> _hi) : lo(_lo), hi(_hi) {}
      CUDAPREFIX Rect(const Rect<DIM>& other) : lo(other.lo), hi(other.hi) {}

      CUDAPREFIX Rect& operator=(const Rect<DIM>& other)
      {
	lo = other.lo;
	hi = other.hi;
	return *this;
      }

      CUDAPREFIX void to_array(coord_t *vals) const { lo.to_array(vals); hi.to_array(vals + DIM); }

      CUDAPREFIX bool operator==(const Rect<DIM>& other)
      {
	return ((lo == other.lo) && (hi == other.hi));
      }

      CUDAPREFIX bool operator!=(const Rect<DIM>& other)
      {
	return ((lo != other.lo) || (hi != other.hi));
      }

      CUDAPREFIX Rect<DIM> operator+(const Point<DIM> &translate) const
      {
        Rect<DIM> result;
        result.lo = lo + translate;
        result.hi = hi + translate;
        return result;
      }

      CUDAPREFIX Rect<DIM> operator-(const Point<DIM> &translate) const
      {
        Rect<DIM> result;
        result.lo = lo - translate;
        result.hi = hi - translate;
        return result;
      }

      CUDAPREFIX Rect<DIM>& operator+=(const Point<DIM> &translate)
      {
        lo += translate;
        hi += translate;
        return *this;
      }

      CUDAPREFIX Rect<DIM>& operator-=(const Point<DIM> &translate)
      {
        lo -= translate;
        hi -= translate;
        return *this;
      }

      CUDAPREFIX bool overlaps(const Rect<DIM>& other) const
      {
	for(unsigned i = 0; i < DIM; i++)
	  if((hi.x[i] < other.lo.x[i]) || (lo.x[i] > other.hi.x[i])) return false;
	return true;
      }

      CUDAPREFIX bool contains(const Rect<DIM>& other) const
      {
	for(unsigned i = 0; i < DIM; i++)
	  if((lo.x[i] > other.lo.x[i]) || (hi.x[i] < other.hi.x[i])) return false;
	return true;
      }

      CUDAPREFIX bool contains(const Point<DIM> &point) const
      {
        for (unsigned i = 0; i < DIM; i++)
          if ((point.x[i] < lo.x[i]) || (point.x[i] > hi.x[i])) return false;
        return true;
      }

      CUDAPREFIX size_t volume(void) const
      {
	size_t v = 1;
	for(unsigned i = 0; i < DIM; i++) {
	  if(lo.x[i] > hi.x[i]) return 0;
	  v *= (hi.x[i] - lo.x[i] + 1);
	}
	return v;
      }

      CUDAPREFIX coord_t dim_size(int dim) const
      {
        assert(dim >= 0);
        assert(dim < int(DIM));
        return (hi.x[dim] - lo.x[dim] + 1);
      }

      CUDAPREFIX Rect<DIM> intersection(const Rect<DIM>& other)
      {
	return Rect<DIM>(Point<DIM>::max(lo, other.lo),
			 Point<DIM>::min(hi, other.hi));
      }

      CUDAPREFIX Rect<DIM> convex_hull(const Rect<DIM>& other)
      {
        return Rect<DIM>(Point<DIM>::min(lo, other.lo),
                         Point<DIM>::max(hi, other.hi));
      }

      bool dominates(const Rect<DIM>& other) const
      {
        if (other.volume() == 0) return true;
        for (unsigned i = 0; i < DIM; i++)
        {
          if (other.lo.x[i] < lo.x[i])
            return false;
          if (other.hi.x[i] > hi.x[i])
            return false;
        }
        return true;
      }
  
      Point<DIM> lo, hi;
    };

    template <unsigned DIM>
    inline std::ostream& operator<<(std::ostream& os, const Rect<DIM>& r)
    {
      os << '[' << r.lo << ',' << r.hi << ']';
      return os;
    }

    template <typename T> class GenericDenseSubrectIterator;
    template <typename T> class GenericLinearSubrectIterator;
    template <unsigned DIM> class GenericPointInRectIterator;

#ifdef OLD_STYLE_DYNAMIC_MAPPINGS  
    template <typename T> class DynamicMapping;
#endif

    template <unsigned IDIM, unsigned ODIM> class Mapping;

#ifdef OLD_STYLE_DYNAMIC_MAPPINGS  
    template <unsigned IDIM, unsigned ODIM>
    class MappingRegistry {
    public:
      typedef Mapping<IDIM, ODIM> *(*MappingDeserializerFn)(const int *data);

      template <typename T>
      bool register_mapping(void) {
	MappingDeserializerFn fnptr = DynamicMapping<T>::deserialize;
	if(fns_to_ids.find(fnptr) != fns_to_ids.end())
	  return false;

	int id = (int)(ids_to_fns.size());
	ids_to_fns.push_back(fnptr);
	fns_to_ids.insert(std::pair<MappingDeserializerFn, int>(fnptr, id));
	return true;
      }

      template <typename T>
      int get_id(void) const {
	MappingDeserializerFn fnptr = T::deserialize;
	typename std::map<MappingDeserializerFn, int>::const_iterator it = fns_to_ids.find(fnptr);
	if(it == fns_to_ids.end())
	  return -1;

	return it->second;
      }

      MappingDeserializerFn get_fnptr(int id) const
      {
	assert((id >= 0) && (id < (int)(ids_to_fns.size())));
	return ids_to_fns[id];
      }

    protected:
      std::vector<MappingDeserializerFn> ids_to_fns;
      std::map<MappingDeserializerFn, int> fns_to_ids;
    };
#endif

    template <unsigned IDIM_, unsigned ODIM_>
    class Mapping {
    private:
      int references;
    public:
      Mapping(void)
        : references(0) { }

      virtual ~Mapping(void) {}
    public:
      static const unsigned IDIM = IDIM_;
      static const unsigned ODIM = ODIM_;

      typedef GenericDenseSubrectIterator<Mapping<IDIM, ODIM> > DenseSubrectIterator;
      typedef GenericLinearSubrectIterator<Mapping<IDIM, ODIM> > LinearSubrectIterator;
      typedef GenericPointInRectIterator<IDIM> PointInInputRectIterator;
      typedef GenericPointInRectIterator<ODIM> PointInOutputRectIterator;

#ifdef OLD_STYLE_DYNAMIC_MAPPINGS  
      static MappingRegistry<IDIM_, ODIM_>& registry(void);

      template <class T>
      static void register_mapping(void)
      {
	registry().template register_mapping<T>();
      }

      static Mapping<IDIM, ODIM> *deserialize_mapping(const int *data)
      {
	typename MappingRegistry<IDIM, ODIM>::MappingDeserializerFn fnptr = registry().get_fnptr(data[0]);
	return (*fnptr)(data);
      }

      virtual void serialize_mapping(int *data) const = 0;

      template <typename T>
      static Mapping<IDIM, ODIM> *new_dynamic_mapping(const T& _t)
      {
	DynamicMapping<T> *m = new DynamicMapping<T>(_t);
	return m;
      }
#endif

      virtual Point<ODIM> image(const Point<IDIM> p) const = 0;
  
      virtual Rect<ODIM> image_convex(const Rect<IDIM> r) const = 0;
      virtual bool image_is_dense(const Rect<IDIM> r) const = 0;
  
      virtual Rect<ODIM> image_dense_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect) const = 0;
      virtual Point<ODIM> image_linear_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect, Point<ODIM> strides[IDIM]) const = 0;

      virtual Rect<IDIM> preimage(const Point<ODIM> p) const = 0;//{ assert(0); return Rect<IDIM>(); }//= 0;
      virtual bool preimage_is_dense(const Point<ODIM> p) const = 0;//{ assert(0); return false; }//= 0;

      inline void add_reference(void)
      {
        __sync_fetch_and_add(&references, 1);
      }
      inline bool remove_reference(void)
      {
        int prev = __sync_fetch_and_sub(&references, 1);
        assert(prev >= 1);
        return (prev == 1);
      }
    };

#ifdef OLD_STYLE_DYNAMIC_MAPPINGS  
    template <typename T>
    class DynamicMapping : public Mapping<T::IDIM, T::ODIM> {
    public:
      enum { IDIM = T::IDIM, ODIM = T::ODIM };
      T t;

      DynamicMapping(void) {}
      DynamicMapping(const T& _t) : t(_t) {}

      virtual void serialize_mapping(int *data) const
      {
	data[0] = Mapping<T::IDIM, T::ODIM>::registry().template get_id<DynamicMapping<T> >();
	memcpy(data + 1, &t, sizeof(T));
      }

      static Mapping<IDIM, ODIM> *deserialize(const int *data)
      {
#ifndef NDEBUG
	int id = 
#endif
          Mapping<T::IDIM, T::ODIM>::registry().template get_id<DynamicMapping<T> >();
	assert(data[0] == id);
	DynamicMapping<T> *m = new DynamicMapping<T>();
	memcpy(&(m->t), data + 1, sizeof(T));
	return m;
      }

      virtual Point<ODIM> image(const Point<IDIM> p) const
      {
	return t.image(p);
      }
  
      virtual Rect<ODIM> image_convex(const Rect<IDIM> r) const
      {
	return t.image_convex(r);
      }

      virtual bool image_is_dense(const Rect<IDIM> r) const
      {
	return t.image_is_dense(r);
      }
  
      virtual Rect<ODIM> image_dense_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect) const
      {
	return t.image_dense_subrect(r, subrect);
      }

      virtual Point<ODIM> image_linear_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect, Point<ODIM> strides[IDIM]) const
      {
	return t.image_linear_subrect(r, subrect, strides);
      }

      virtual Rect<IDIM> preimage(const Point<ODIM> p) const
      {
	return t.preimage(p);
      }

      virtual bool preimage_is_dense(const Point<ODIM> p) const
      {
	return t.preimage_is_dense(p);
      }
    };
#endif

    template <unsigned DIM>
    class GenericPointInRectIterator {
    public:
      GenericPointInRectIterator(const Rect<DIM> _r) : r(_r)
      {
	p = r.lo;
	any_left = (r.lo <= r.hi);
      }

      Rect<DIM> r;
      Point<DIM> p;
      bool any_left;
      
      bool step(void)
      {
	for(unsigned i = 0; i < DIM; i++) {
	  p.x[i]++;
	  if(p.x[i] <= r.hi.x[i]) return true;
	  p.x[i] = r.lo.x[i];
	}
	// if we fall through, we've exhausted all the dimensions
	any_left = false;
	return false;
      }

      operator bool(void) const { return any_left; }
      GenericPointInRectIterator& operator++(/*i am prefix*/) { step(); return *this; }
      GenericPointInRectIterator operator++(int /*i am postfix*/)
      { 
	GenericPointInRectIterator<DIM> orig = *this; 
	step();
	return orig;
      }
    };
  
    template <typename T>
    class GenericDenseSubrectIterator {
    public:
      GenericDenseSubrectIterator(const Rect<T::IDIM> r, const T& m)
	: orig_rect(r), mapping(m)
      {
	image = m.image_dense_subrect(r, subrect);
        any_left = true;
      }

      Rect<T::IDIM> orig_rect;
      const T& mapping;
      Rect<T::ODIM> image;
      Rect<T::IDIM> subrect;
      bool any_left;
      
      bool step(void)
      {
	// to make a step, find the "seam" along which we split last time - first dimension whose
	//  subrect.hi isn't the edge of the original rect
	unsigned seam_idx = 0;
	while(subrect.hi.x[seam_idx] == orig_rect.hi.x[seam_idx]) {
	  seam_idx++;
          if(seam_idx >= T::IDIM) {
            any_left = false;
            return false;
          }
	}
	// ask for the rest of the original rect along the current split
	Rect<T::IDIM> newrect;
	// dimensions below the seam use the original rectangle bounds
	for(unsigned i = 0; i < seam_idx; i++) {
	  newrect.lo.x[i] = orig_rect.lo.x[i];
	  newrect.hi.x[i] = orig_rect.hi.x[i];
	}
	// the seam continues where we left off
	newrect.lo.x[seam_idx] = subrect.hi.x[seam_idx] + 1;
	newrect.hi.x[seam_idx] = orig_rect.hi.x[seam_idx];
	// above the seam tries to use the same cross section
	for(unsigned i = seam_idx + 1; i < T::IDIM; i++) {
	  newrect.lo.x[i] = subrect.lo.x[i];
	  newrect.hi.x[i] = subrect.hi.x[i];
	}

	image = mapping.image_dense_subrect(newrect, subrect);

	// sanity check that dimensions above the current seam didn't further split
	for(unsigned i = seam_idx + 1; i < T::IDIM; i++) {
	  assert(newrect.lo.x[i] == subrect.lo.x[i]);
	  assert(newrect.hi.x[i] == subrect.hi.x[i]);
	}

	return true;
      }

      operator bool(void) const { return any_left; }
      GenericDenseSubrectIterator& operator++(int /*i am postfix*/) { step(); return *this; }
    };
  
    template <typename T>
    class GenericLinearSubrectIterator {
    public:
      GenericLinearSubrectIterator(const Rect<T::IDIM> r, const T& m)
	: orig_rect(r), mapping(m)
      {
	image_lo = m.image_linear_subrect(r, subrect, strides);
	any_left = true;
      }

      Rect<T::IDIM> orig_rect;
      const T& mapping;
      Point<T::ODIM> image_lo;
      Rect<T::IDIM> subrect;
      Point<T::ODIM> strides[T::IDIM];
      bool any_left;
      
      bool step(void)
      {
	assert(subrect == orig_rect);
	any_left = false;
	return false;
      }

      operator bool(void) const { return any_left; }
      GenericLinearSubrectIterator& operator++(int /*i am postfix*/) { step(); return *this; }
    };
  
    template <unsigned DIM>
    class Translation {
    public:
      enum { IDIM = DIM, ODIM = DIM };
      Translation(void) : offset(coord_t(0)) {}
      Translation(const Point<DIM> _offset) : offset(_offset) {}
  
      Point<ODIM> image(const Point<IDIM> p) const
      {
        return p + offset;
      }
  
      Rect<ODIM> image_convex(const Rect<IDIM> r) const
      {
        return Rect<ODIM>(r.lo + offset, r.hi + offset);
      }
  
      bool image_is_dense(const Rect<IDIM> r) const
      {
        return true;
      }
      
      Rect<ODIM> image_dense_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect) const
      {
	subrect = r;
	return Rect<ODIM>(r.lo + offset, r.hi + offset);
      }

      Point<ODIM> image_linear_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect, Point<ODIM> strides[IDIM]) const
      {
	subrect = r;
	for(unsigned i = 0; i < DIM; i++) {
	  // strides are unit vectors
	  strides[i] = coord_t(0);
	  strides[i].x[i] = 1;
	}
	return r.lo + offset;
      }

      Rect<IDIM> preimage(const Point<ODIM> p) const
      {
	return Rect<IDIM>(p - offset, p - offset);
      }

      bool preimage_is_dense(const Point<ODIM> p) const
      {
	return true;
      }

    protected:
      Point<DIM> offset;
    };
  
    template <unsigned DIM>
    class Linearization {
    protected:
      Linearization(void) {}
      Point<DIM> strides;
      coord_t offset;
    public:
      enum { IDIM = DIM, ODIM = 1 };
      typedef GenericDenseSubrectIterator<Linearization<DIM> > DenseSubrectIterator;
      typedef GenericLinearSubrectIterator<Linearization<DIM> > LinearSubrectIterator;
      typedef GenericPointInRectIterator<IDIM> PointInInputRectIterator;
      typedef GenericPointInRectIterator<ODIM> PointInOutputRectIterator;

      Linearization(const Point<DIM> _strides, coord_t _offset = 0)
        : strides(_strides), offset(_offset) {}

      Point<1> image(const Point<IDIM> p) const
      {
        return p.dot(strides) + offset;
      }
  
      Rect<1> image_convex(const Rect<IDIM> r) const
      {
        coord_t lo = offset;
        coord_t hi = offset;
        for(int i = 0; i < IDIM; i++)
  	if(strides[i] > 0) {
  	  lo += strides[i] * r.lo[i];
  	  hi += strides[i] * r.hi[i];
  	} else {
  	  lo += strides[i] * r.hi[i];
  	  hi += strides[i] * r.lo[i];
  	}
        return Rect<1>(lo, hi);
      }
  
      bool image_is_dense(const Rect<IDIM> r) const
      {
        // not the most efficient, but should work: see if size of convex image is product of dimensions
        Rect<1> convex = image_convex(r);
        coord_t prod = 1;
        for(int i = 0; i < IDIM; i++) {
  	prod *= 1 + (r.hi[i] - r.lo[i]);
        }
        return (convex.hi[0] - convex.lo[0] + 1) == prod;
      }

      Rect<ODIM> image_dense_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect) const
      {
	coord_t count = 1;
	Rect<IDIM> s(r.lo, r.lo);
	for(unsigned i = 0; i < IDIM; i++) {
	  bool found = false;
	  for(unsigned j = 0; j < IDIM; j++) {
	    if(strides[j] == count) {
	      s.hi.x[j] = r.hi.x[j];
	      count *= (r.hi.x[j] - r.lo.x[j] + 1);
	      found = true;
	      break;
	    }
	  }
	  if(!found) break;
	}
	assert(image_is_dense(s));
	subrect = s;
	return image_convex(s);
      }

      Point<ODIM> image_linear_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect, Point<ODIM> strides[IDIM]) const
      {
	// linearization is always, well, linear
	subrect = r;
	for(unsigned i = 0; i < IDIM; i++)
	  strides[i] = this->strides[i];
	return image(r.lo);
      }

      Rect<IDIM> preimage(const Point<ODIM> p) const
      {
	// we only have enough information to do this in the 1-D case
	if(IDIM == 1) {
	  coord_t delta = p[0] - offset;
	  Point<IDIM> lo, hi;
	  // stride must divide evenly, otherwise p has no preimage
	  if(strides[0] == 1) { // optimize for common divide-by-1 case
	    lo.x[0] = delta;
	    hi = lo;
	  } else if((delta % strides[0]) == 0) {
	    lo.x[0] = delta / strides[0];
	    hi = lo;
	  } else {
	    lo.x[0] = 0;
	    hi.x[0] = -1; // hi < lo means empty rectangle
	  }
	  return Rect<IDIM>(lo, hi);
	} else {
	  assert(0);
	  return Rect<IDIM>();
	}
      }

      bool preimage_is_dense(const Point<ODIM> p) const
      {
	return true;
      }
    };

    template <unsigned DIM>
    class CArrayLinearization : public Linearization<DIM> {
    public:
      CArrayLinearization(void) {}
      CArrayLinearization(Rect<DIM> bounds, coord_t first_index = 0)
      {
	Linearization<DIM>::strides.x[DIM - 1] = 1;
	for(int i = int(DIM) - 2; i >= 0; i--)
	  this->strides.x[i] = this->strides.x[i + 1] * (bounds.hi[i + 1] - bounds.lo[i + 1] + 1);
	this->offset = first_index - this->strides.dot(bounds.lo);
      }
    };

    template <unsigned DIM>
    class FortranArrayLinearization : public Linearization<DIM> {
    public:
      FortranArrayLinearization(void) {}
      FortranArrayLinearization(Rect<DIM> bounds, coord_t first_index = 1)
      {
	this->strides.x[0] = 1;
	for(unsigned i = 1; i < DIM; i++)
	  this->strides.x[i] = this->strides.x[i - 1] * (bounds.hi[i - 1] - bounds.lo[i - 1] + 1);
	this->offset = first_index - this->strides.dot(bounds.lo);
      }
    };

    template <unsigned DIM>
    class Blockify {
    public:
      enum { IDIM = DIM, ODIM = DIM };
      typedef GenericDenseSubrectIterator<Blockify<DIM> > DenseSubrectIterator;
      typedef GenericLinearSubrectIterator<Blockify<DIM> > LinearSubrectIterator;
      typedef GenericPointInRectIterator<DIM> PointInInputRectIterator;
      typedef GenericPointInRectIterator<DIM> PointInOutputRectIterator;

      Blockify(void) {}
      Blockify(Point<DIM> _block_size) :
        block_size(_block_size),
        offset(Point<DIM>::ZEROES())
      {
      }

      Blockify(Point<DIM> _block_size, Point<DIM> _offset) :
        block_size(_block_size),
        offset(_offset)
      {
      }

      Point<DIM> image(const Point<DIM> p) const
      {
	Point<DIM> q;
	for(unsigned i = 0; i < DIM; i++)
	  q.x[i] = (p.x[i] - offset.x[i]) / block_size.x[i];
	return q;
      }
  
      Rect<DIM> image_convex(const Rect<DIM> r) const
      {
	return Rect<DIM>(image(r.lo),
			 image(r.hi));
      }

      bool image_is_dense(const Rect<DIM> r) const
      {
	return true;
      }
  
      Rect<DIM> image_dense_subrect(const Rect<DIM> r, Rect<DIM>& subrect) const
      {
	subrect = r;
	return image_convex(r);
      }

      Point<DIM> image_linear_subrect(const Rect<DIM> r, Rect<DIM>& subrect, Point<DIM> strides[DIM]) const
      {
	assert(0);
        return Point<DIM>();
      }

      Rect<DIM> preimage(const Point<DIM> p) const
      {
	Rect<DIM> q;
	for(unsigned i = 0; i < DIM; i++) {
	  q.lo.x[i] = p.x[i] * block_size.x[i] + offset[i];
	  q.hi.x[i] = q.lo.x[i] + (block_size.x[i] - 1);
	}
	return q;
      }

      bool preimage_is_dense(const Point<DIM> p) const
      {
	return true;
      }

    protected:
      Point<DIM> block_size;
      Point<DIM> offset;
    };
  }; // namespace Arrays
}; // namespace LegionRuntime

#endif
