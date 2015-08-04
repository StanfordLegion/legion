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

// stuff to try making array accesses more sane

#ifndef _ARRAYS_H
#define _ARRAYS_H

#include <vector>
#include <map>

#include <cassert>
#ifndef __GNUC__
#include "atomics.h" // for __sync_fetch_and_add
#endif

// why must I define this every time I need it?
static inline int imin(int a, int b) { return (a < b) ? a : b; }
static inline int imax(int a, int b) { return (a > b) ? a : b; }

namespace LegionRuntime {
  namespace Arrays {
    template <unsigned DIM>
    class Point {
    public:
      Point(void) {}
      Point(const int *vals) { for(unsigned i = 0; i < DIM; i++) x[i] = vals[i]; }
      Point(const Point<DIM>& other) { for(unsigned i = 0; i < DIM; i++) x[i] = other.x[i]; }

      Point& operator=(const Point<DIM>& other) 
      { 
	for(unsigned i = 0; i < DIM; i++) x[i] = other.x[i];
	return *this;
      }

      void to_array(int *vals) const { for(unsigned i = 0; i < DIM; i++) vals[i] = x[i]; }

      bool operator==(const Point<DIM>& other) const
      { 
	for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] != other.x[i]) return false; 
	return true; 
      }

      bool operator!=(const Point<DIM>& other) const
      { 
	for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] != other.x[i]) return true; 
	return false; 
      }

      bool operator<=(const Point<DIM>& other) const
      { 
	for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] > other.x[i]) return false; 
	return true; 
      }

      struct STLComparator {
	bool operator()(const Point<DIM>& a, const Point<DIM>& b) const
	{
	  for(unsigned i = 0; i < DIM; i++)  {
	    int d = a.x[i] - b.x[i];
	    if (d < 0) return true;
	    if (d > 0) return false;
	  }
	  return false;
	}
      };

      int operator[](unsigned idx) const { return x[idx]; }
  
      static Point<DIM> ZEROES(void)
      {
        Point<DIM> z; for(unsigned i = 0; i < DIM; i++) z.x[i] = 0; return z;
      }

      static Point<DIM> ONES(void)
      {
        Point<DIM> o; for(unsigned i = 0; i < DIM; i++) o.x[i] = 1; return o;
      }

      Point<DIM> operator+(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
          res.x[i] = x[i] + other.x[i];
	return res;
      }
  
      Point<DIM> operator-(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] - other.x[i];
	return res;
      }
  
      // element-wise multiplication and division
      Point<DIM> operator*(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] * other.x[i];
	return res;
      }
  
      Point<DIM> operator/(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] / other.x[i];
	return res;
      }

      static Point<DIM> sum(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = a.x[i] + b.x[i];
	return res;
      }

      static Point<DIM> min(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = imin(a.x[i], b.x[i]);
	return res;
      }

      static Point<DIM> max(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = imax(a.x[i], b.x[i]);
	return res;
      }

      static int dot(const Point<DIM> a, const Point<DIM> b)
      {
	int v = 0;
        for(unsigned i = 0; i < DIM; i++)
	  v += a.x[i] * b.x[i];
	return v;
      }
  
      int dot(const Point<DIM> other) const
      {
        int v = 0;
        for(unsigned i = 0; i < DIM; i++) v += x[i] * other.x[i];
        return v;
      }
  
    public:
      int x[DIM];
    };
  
    template <>
    class Point<1> {
    public:
      enum { DIM = 1 };
      Point(void) {}
      Point(int val) { x[0] = val; }
      Point(const int *vals) { for(unsigned i = 0; i < DIM; i++) x[i] = vals[i]; }
      Point(const Point<1>& other) { for(unsigned i = 0; i < DIM; i++) x[i] = other.x[i]; }

      Point& operator=(const Point<1>& other) 
      { 
	for(unsigned i = 0; i < 1; i++) x[i] = other.x[i];
	return *this;
      }

      void to_array(int *vals) const { for(unsigned i = 0; i < DIM; i++) vals[i] = x[i]; }
  
      int operator[](unsigned idx) const { return x[0]; }
      operator int(void) const { return x[0]; }
      
      bool operator==(const Point<DIM> &other) const
      {
        for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] != other.x[i]) return false; 
	return true;
      }

      bool operator!=(const Point<DIM> &other) const
      {
        for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] != other.x[i]) return true; 
	return false;
      }

      bool operator<=(const Point<DIM> &other) const
      {
        for(unsigned i = 0; i < DIM; i++) 
	  if(x[i] > other.x[i]) return false; 
	return true;
      }
  
      static Point<DIM> ZEROES(void)
      {
        Point<DIM> z; for(unsigned i = 0; i < DIM; i++) z.x[i] = 0; return z;
      }

      static Point<DIM> ONES(void)
      {
        Point<DIM> o; for(unsigned i = 0; i < DIM; i++) o.x[i] = 1; return o;
      }

      static Point<DIM> sum(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = a.x[i] + b.x[i];
	return res;
      }

      static Point<DIM> min(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = imin(a.x[i], b.x[i]);
	return res;
      }

      static Point<DIM> max(const Point<DIM> a, const Point<DIM> b)
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = imax(a.x[i], b.x[i]);
	return res;
      }

      static int dot(const Point<DIM> a, const Point<DIM> b)
      {
	int v = 0;
        for(unsigned i = 0; i < DIM; i++)
	  v += a.x[i] * b.x[i];
	return v;
      }

      Point<DIM> operator+(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
          res.x[i] = x[i] + other.x[i];
	return res;
      }

      Point<DIM> operator-(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] - other.x[i];
	return res;
      }
  
      // element-wise multiplication and division
      Point<DIM> operator*(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] * other.x[i];
	return res;
      }
  
      Point<DIM> operator/(const Point<DIM> other) const
      {
        Point<DIM> res;
        for(unsigned i = 0; i < DIM; i++)
	  res.x[i] = x[i] / other.x[i];
	return res;
      }

      int dot(const Point<DIM> other) const { return dot(*this, other); }
  
    public:
      int x[1];
    };

    inline Point<1> make_point(int x)
    {
      Point<1> p;
      p.x[0] = x;
      return p;
    }

    inline Point<2> make_point(int x, int y)
    {
      Point<2> p;
      p.x[0] = x;
      p.x[1] = y;
      return p;
    }

    inline Point<3> make_point(int x, int y, int z)
    {
      Point<3> p;
      p.x[0] = x;
      p.x[1] = y;
      p.x[2] = z;
      return p;
    }

    template <unsigned DIM>
    class Rect {
    public:
      Rect(void) {}
      explicit Rect(const int *vals) : lo(vals), hi(vals + DIM) {}
      Rect(const Point<DIM> _lo, const Point<DIM> _hi) : lo(_lo), hi(_hi) {}
      Rect(const Rect<DIM>& other) : lo(other.lo), hi(other.hi) {}

      Rect& operator=(const Rect<DIM>& other)
      {
	lo = other.lo;
	hi = other.hi;
	return *this;
      }

      void to_array(int *vals) const { lo.to_array(vals); hi.to_array(vals + DIM); }

      bool operator==(const Rect<DIM>& other)
      {
	return ((lo == other.lo) && (hi == other.hi));
      }

      bool operator!=(const Rect<DIM>& other)
      {
	return ((lo != other.lo) || (hi != other.hi));
      }

      bool overlaps(const Rect<DIM>& other) const
      {
	for(unsigned i = 0; i < DIM; i++)
	  if((hi.x[i] < other.lo.x[i]) || (lo.x[i] > other.hi.x[i])) return false;
	return true;
      }

      bool contains(const Rect<DIM>& other) const
      {
	for(unsigned i = 0; i < DIM; i++)
	  if((lo.x[i] > other.lo.x[i]) || (hi.x[i] < other.hi.x[i])) return false;
	return true;
      }

      bool contains(const Point<DIM> &point) const
      {
        for (unsigned i = 0; i < DIM; i++)
          if ((point.x[i] < lo.x[i]) || (point.x[i] > hi.x[i])) return false;
        return true;
      }

      size_t volume(void) const
      {
	size_t v = 1;
	for(unsigned i = 0; i < DIM; i++) {
	  if(lo.x[i] > hi.x[i]) return 0;
	  v *= (hi.x[i] - lo.x[i] + 1);
	}
	return v;
      }

      int dim_size(int dim) const
      {
        assert(dim >= 0);
        assert(dim < int(DIM));
        return (hi.x[dim] - lo.x[dim] + 1);
      }

      Rect<DIM> intersection(const Rect<DIM>& other)
      {
	return Rect<DIM>(Point<DIM>::max(lo, other.lo),
			 Point<DIM>::min(hi, other.hi));
      }

      Rect<DIM> convex_hull(const Rect<DIM>& other)
      {
        return Rect<DIM>(Point<DIM>::min(lo, other.lo),
                         Point<DIM>::max(hi, other.hi));
      }
  
      Point<DIM> lo, hi;
    };

    template <typename T> class GenericDenseSubrectIterator;
    template <typename T> class GenericLinearSubrectIterator;
    template <unsigned DIM> class GenericPointInRectIterator;
  
    template <typename T> class DynamicMapping;

    template <unsigned IDIM, unsigned ODIM> class Mapping;

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

      static MappingRegistry<IDIM_, ODIM_> registry;

      template <class T>
      static void register_mapping(void)
      {
	registry.template register_mapping<T>();
      }

      static Mapping<IDIM, ODIM> *deserialize_mapping(const int *data)
      {
	typename MappingRegistry<IDIM, ODIM>::MappingDeserializerFn fnptr = registry.get_fnptr(data[0]);
	return (*fnptr)(data);
      }

      virtual void serialize_mapping(int *data) const = 0;

      template <typename T>
      static Mapping<IDIM, ODIM> *new_dynamic_mapping(const T& _t)
      {
	DynamicMapping<T> *m = new DynamicMapping<T>(_t);
	return m;
      }

      virtual Point<ODIM> image(const Point<IDIM> p) const = 0;
  
      virtual Rect<ODIM> image_convex(const Rect<IDIM> r) const = 0;
      virtual bool image_is_dense(const Rect<IDIM> r) const = 0;
  
      virtual Rect<ODIM> image_dense_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect) const = 0;
      virtual Point<ODIM> image_linear_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect, Point<ODIM> strides[IDIM]) const = 0;

      virtual Rect<IDIM> preimage(const Point<ODIM> p) const { assert(0); return Rect<IDIM>(); }//= 0;
      virtual bool preimage_is_dense(const Point<ODIM> p) const { assert(0); return false; }//= 0;

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

    template <typename T>
    class DynamicMapping : public Mapping<T::IDIM, T::ODIM> {
    public:
      enum { IDIM = T::IDIM, ODIM = T::ODIM };
      T t;

      DynamicMapping(void) {}
      DynamicMapping(const T& _t) : t(_t) {}

      virtual void serialize_mapping(int *data) const
      {
	data[0] = Mapping<T::IDIM, T::ODIM>::registry.template get_id<DynamicMapping<T> >();
	memcpy(data + 1, &t, sizeof(T));
      }

      static Mapping<IDIM, ODIM> *deserialize(const int *data)
      {
#ifndef NDEBUG
	int id = 
#endif
          Mapping<T::IDIM, T::ODIM>::registry.template get_id<DynamicMapping<T> >();
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
      GenericPointInRectIterator& operator++(int /*i am postfix*/) { step(); return *this; }
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
      Translation(void) : offset(0) {}
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
	  strides[i] = 0;
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
      int offset;
    public:
      enum { IDIM = DIM, ODIM = 1 };
      typedef GenericDenseSubrectIterator<Linearization<DIM> > DenseSubrectIterator;
      typedef GenericLinearSubrectIterator<Linearization<DIM> > LinearSubrectIterator;
      typedef GenericPointInRectIterator<IDIM> PointInInputRectIterator;
      typedef GenericPointInRectIterator<ODIM> PointInOutputRectIterator;

      Linearization(const Point<DIM> _strides, int _offset = 0)
        : strides(_strides), offset(_offset) {}

      Point<1> image(const Point<IDIM> p) const
      {
        return p.dot(strides) + offset;
      }
  
      Rect<1> image_convex(const Rect<IDIM> r) const
      {
        int lo = offset;
        int hi = offset;
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
        int prod = 1;
        for(int i = 0; i < IDIM; i++) {
  	prod *= 1 + (r.hi[i] - r.lo[i]);
        }
        return (convex.hi[0] - convex.lo[0] + 1) == prod;
      }

      Rect<ODIM> image_dense_subrect(const Rect<IDIM> r, Rect<IDIM>& subrect) const
      {
	int count = 1;
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
	assert(0);
	return Rect<IDIM>();
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
      CArrayLinearization(Rect<DIM> bounds, int first_index = 0)
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
      FortranArrayLinearization(Rect<DIM> bounds, int first_index = 1)
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
      Blockify(Point<DIM> _block_size) : block_size(_block_size) {}

      Point<DIM> image(const Point<DIM> p) const
      {
	Point<DIM> q;
	for(unsigned i = 0; i < DIM; i++)
	  q.x[i] = p.x[i] / block_size.x[i];
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
	  q.lo.x[i] = p.x[i] * block_size.x[i];
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
    };
  }; // namespace Arrays
}; // namespace LegionRuntime

#endif
