/* Copyright 2013 Stanford University
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

#ifndef RUNTIME_ACCESSOR_H
#define RUNTIME_ACCESSOR_H

#ifdef __CUDACC__
#define CUDAPREFIX __host__ __device__
#else
#define CUDAPREFIX
#endif

#include "arrays.h"

#ifndef __GNUC__
#include "atomics.h" // for __sync_fetch_and_add
#endif

using namespace LegionRuntime::Arrays;

namespace LegionRuntime {
#ifdef PRIVILEGE_CHECKS
  enum AccessorPrivilege {
    ACCESSOR_NONE   = 0x00000000,
    ACCESSOR_READ   = 0x00000001,
    ACCESSOR_WRITE  = 0x00000002,
    ACCESSOR_REDUCE = 0x00000004,
    ACCESSOR_ALL    = 0x00000007,
  };
#endif
  namespace LowLevel {
    class DomainPoint;
    class Domain;
  }

  namespace Accessor {

    class ByteOffset {
    public:
      ByteOffset(void) : offset(0) {}
      explicit ByteOffset(off_t _offset) : offset(_offset) { assert(offset == _offset); }
      explicit ByteOffset(int _offset) : offset(_offset) {}

      template <typename T1, typename T2>
      ByteOffset(T1 *p1, T2 *p2) : offset(((char *)p1) - ((char *)p2)) {}

      template <typename T>
      T *add_to_pointer(T *ptr) const
      {
	return (T*)(((char *)ptr) + offset);
      }

      bool operator==(const ByteOffset rhs) const { return offset == rhs.offset; }
      bool operator!=(const ByteOffset rhs) const { return offset != rhs.offset; }

      ByteOffset& operator+=(ByteOffset rhs) { offset += rhs.offset; return *this; }
      ByteOffset& operator*=(int scale) { offset *= scale; return *this; }
      
      ByteOffset operator+(const ByteOffset rhs) const { return ByteOffset(offset + rhs.offset); }
      ByteOffset operator*(int scale) const { return ByteOffset(offset * scale); }
      
      int offset;
    };

    inline ByteOffset operator*(int scale, const ByteOffset rhs) { return ByteOffset(scale * rhs.offset); }

    template <typename T>
    inline T* operator+(T *ptr, const ByteOffset offset) { return offset.add_to_pointer(ptr); }

    template <typename T>
    inline T*& operator+=(T *&ptr, const ByteOffset offset) { ptr = offset.add_to_pointer(ptr); return ptr; }

    namespace TemplateFu {
      // bool IsAStruct<T>::value == true if T is a class/struct, else false
      template <typename T> 
      struct IsAStruct {
	typedef char yes[2];
	typedef char no[1];

	template <typename T2> static yes& test_for_struct(int T2:: *x);
	template <typename T2> static no& test_for_struct(...);

	static const bool value = sizeof(test_for_struct<T>(0)) == sizeof(yes);
      };
    };

    template <typename AT, typename ET = void, typename PT = ET> struct RegionAccessor;

    template <typename AT> struct RegionAccessor<AT, void, void> : public AT::Untyped {
      CUDAPREFIX
      RegionAccessor(const typename AT::Untyped& to_copy)
	: AT::Untyped(to_copy) {}
    };

    // Helper function for extracting an accessor from a region
    template<typename AT, typename ET, typename RT>
    inline RegionAccessor<AT,ET> extract_accessor(const RT &region) 
    {
      return region.get_accessor().template typeify<ET>().template convert<AT>();
    }

    namespace AccessorType {
      template <typename T, off_t val> 
      struct Const {
      public:
        CUDAPREFIX
        Const(void) {}
        CUDAPREFIX
        Const(T _value) { assert(_value == val); }
	static const T value = val;
      };

      template <typename T> 
      struct Const<T, 0> {
      public:
        CUDAPREFIX
        Const(void) : value(0) {}
        CUDAPREFIX
        Const(T _value) : value(_value) {}
	const T value;
      };

      template <size_t STRIDE> struct AOS;
      template <size_t STRIDE> struct SOA;
      template <size_t STRIDE, size_t BLOCK_SIZE, size_t BLOCK_STRIDE> struct HybridSOA;

      template <typename REDOP> struct ReductionFold;
      template <typename REDOP> struct ReductionList;

#ifdef POINTER_CHECKS
      // TODO: Make this function work for GPUs
      void verify_access(void *impl_ptr, unsigned ptr);
      void verify_access(void *impl_ptr, const LowLevel::DomainPoint& dp);
#endif

      struct Generic {
	struct Untyped {
	  Untyped() : internal(0), field_offset(0) {}
	  Untyped(void *_internal, off_t _field_offset = 0) : internal(_internal), field_offset(_field_offset) {}

	  template <typename ET>
	  RegionAccessor<Generic, ET, ET> typeify(void) const {
	    RegionAccessor<Generic, ET, ET> result(Typed<ET, ET>(internal, field_offset));
#ifdef PRIVILEGE_CHECKS
            result.set_privileges(priv);
#endif
            return result;
	  }

	  void read_untyped(ptr_t ptr, void *dst, size_t bytes, off_t offset = 0) const;
	  void write_untyped(ptr_t ptr, const void *src, size_t bytes, off_t offset = 0) const;

	  void read_untyped(const LowLevel::DomainPoint& dp, void *dst, size_t bytes, off_t offset = 0) const;
	  void write_untyped(const LowLevel::DomainPoint& dp, const void *src, size_t bytes, off_t offset = 0) const;

	  RegionAccessor<Generic, void, void> get_untyped_field_accessor(off_t _field_offset, size_t _field_size)
	  {
	    return RegionAccessor<Generic, void, void>(Untyped(internal, field_offset + _field_offset));
	  }

	  template <int DIM>
	  void *raw_rect_ptr(const Rect<DIM>& r, Rect<DIM> &subrect, ByteOffset *offsets);

	  template <int DIM>
	  void *raw_rect_ptr(const Rect<DIM>& r, Rect<DIM> &subrect, ByteOffset *offsets,
			     const std::vector<off_t> &field_offsets, ByteOffset &field_stride);

	  template <int DIM>
	  void *raw_dense_ptr(const Rect<DIM>& r, Rect<DIM> &subrect, ByteOffset &elem_stride);

	  template <int DIM>
	  void *raw_dense_ptr(const Rect<DIM>& r, Rect<DIM> &subrect, ByteOffset &elem_stride,
			      const std::vector<off_t> &field_offsets, ByteOffset &field_stride);

	  void *internal;
	  off_t field_offset;
#ifdef PRIVILEGE_CHECKS
        protected:
          AccessorPrivilege priv;
        public:
          inline void set_privileges_untyped(AccessorPrivilege p) { priv = p; }
#endif
	protected:
	  bool get_aos_parameters(void *& base, size_t& stride) const;
	  bool get_soa_parameters(void *& base, size_t& stride) const;
	  bool get_hybrid_soa_parameters(void *& base, size_t& stride,
					 size_t& block_size, size_t& block_stride) const;
	  bool get_redfold_parameters(void *& base) const;
	  bool get_redlist_parameters(void *& base, ptr_t *& next_ptr) const;
	};

	// empty class that will have stuff put in it later if T is a struct
	template <typename T, typename PT, bool B> struct StructSpecific {};

	template <typename T, typename PT>
	struct Typed : public Untyped, public StructSpecific<T, PT, TemplateFu::IsAStruct<T>::value> {
	  Typed() : Untyped() {}
	  Typed(void *_internal, off_t _field_offset = 0) : Untyped(_internal, _field_offset) {}

#ifdef PRIVILEGE_CHECKS
          inline void set_privileges(AccessorPrivilege p) { priv = p; }
#endif

	  template <typename PTRTYPE>
	  inline T read(PTRTYPE ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(priv & ACCESSOR_READ);
#endif
#ifdef POINTER_CHECKS
            verify_access(internal, ptr);
#endif
            T val; read_untyped(ptr, &val, sizeof(val)); return val; 
          }

	  template <typename PTRTYPE>
	  inline void write(PTRTYPE ptr, const T& newval) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(internal, ptr);
#endif
            write_untyped(ptr, &newval, sizeof(newval)); 
          }

	  template <int DIM>
	  T *raw_rect_ptr(const Rect<DIM>& r, Rect<DIM> &subrect, ByteOffset *offsets)
	  { return (T*)(Untyped::raw_rect_ptr<DIM>(r, subrect, offsets)); }

	  template <int DIM>
	    T *raw_rect_ptr(const Rect<DIM>& r, Rect<DIM> &subrect, ByteOffset *offsets,
			    const std::vector<off_t> &field_offsets, ByteOffset &field_stride)
	  { return (T*)(Untyped::raw_rect_ptr<DIM>(r, subrect, offsets, field_offsets, field_stride)); }

	  template <int DIM>
	  T *raw_dense_ptr(const Rect<DIM>& r, Rect<DIM> &subrect, ByteOffset &elem_stride)
	  { return (T*)(Untyped::raw_dense_ptr<DIM>(r, subrect, elem_stride)); }

	  template<typename REDOP>
	  inline void reduce(ptr_t ptr, typename REDOP::RHS newval) const
	  {
#ifdef PRIVILEGE_CHECKS
            assert(priv & ACCESSOR_REDUCE);
#endif
#ifdef POINTER_CHECKS
	    verify_access(internal, ptr.value);
#endif
	    T val = read(ptr);
	    REDOP::template apply<true>(val, newval);
	    write(ptr, val);
	  }

	  typedef AOS<sizeof(PT)> AOS_TYPE;

	  template <typename AT>
	  bool can_convert(void) const {
	    return can_convert_helper<AT>(static_cast<AT *>(0));
	  }

	  template <typename AT>
	  RegionAccessor<AT, T> convert(void) const {
	    return convert_helper<AT>(static_cast<AT *>(0));
	  }

	  template <typename AT, size_t STRIDE>
	  bool can_convert_helper(AOS<STRIDE> *dummy) const {
	    //printf("in aos(%zd) converter\n", STRIDE);
	    void *aos_base = 0;
	    size_t aos_stride = STRIDE;
	    bool ok = get_aos_parameters(aos_base, aos_stride);
	    return ok;
	  }

	  template <typename AT, size_t STRIDE>
	  RegionAccessor<AT, T> convert_helper(AOS<STRIDE> *dummy) const {
	    //printf("in aos(%zd) converter\n", STRIDE);
	    void *aos_base = 0;
	    size_t aos_stride = STRIDE;
	    bool ok = get_aos_parameters(aos_base, aos_stride);
	    assert(ok);
	    typename AT::template Typed<T, T> t(aos_base, aos_stride);
            RegionAccessor<AT, T> result(t);
#ifdef POINTER_CHECKS
            result.set_impl(internal);
#endif
#ifdef PRIVILEGE_CHECKS
            result.set_privileges(priv);
#endif
            return result;
	  }
	      
	  template <typename AT, size_t STRIDE>
	  bool can_convert_helper(SOA<STRIDE> *dummy) const {
	    //printf("in soa(%zd) converter\n", STRIDE);
	    void *soa_base = 0;
	    size_t soa_stride = STRIDE;
	    bool ok = get_soa_parameters(soa_base, soa_stride);
	    return ok;
	  }

	  template <typename AT, size_t STRIDE>
	  RegionAccessor<AT, T> convert_helper(SOA<STRIDE> *dummy) const {
	    //printf("in soa(%zd) converter\n", STRIDE);
	    void *soa_base = 0;
	    size_t soa_stride = STRIDE;
	    bool ok = get_soa_parameters(soa_base, soa_stride);
	    assert(ok);
	    typename AT::template Typed<T, T> t(soa_base, soa_stride);
            RegionAccessor<AT,T> result(t);
#ifdef POINTER_CHECKS
            result.set_impl(internal);
#endif
#ifdef PRIVILEGE_CHECKS
            result.set_privileges(priv);
#endif
            return result;
	  }
	      
	  template <typename AT, size_t STRIDE, size_t BLOCK_SIZE, size_t BLOCK_STRIDE>
	  bool can_convert_helper(HybridSOA<STRIDE, BLOCK_SIZE, BLOCK_STRIDE> *dummy) const {
	    //printf("in hybridsoa(%zd,%zd,%zd) converter\n", STRIDE, BLOCK_SIZE, BLOCK_STRIDE);
	    void *hybrid_soa_base = 0;
	    size_t hybrid_soa_stride = STRIDE;
	    size_t hybrid_soa_block_size = BLOCK_SIZE;
	    size_t hybrid_soa_block_stride = BLOCK_STRIDE;
	    bool ok = get_hybrid_soa_parameters(hybrid_soa_base, hybrid_soa_stride,
						hybrid_soa_block_size, hybrid_soa_block_stride);
	    return ok;
	  }

	  template <typename AT, size_t STRIDE, size_t BLOCK_SIZE, size_t BLOCK_STRIDE>
	  RegionAccessor<AT, T> convert_helper(HybridSOA<STRIDE, BLOCK_SIZE, BLOCK_STRIDE> *dummy) const {
	    //printf("in hybridsoa(%zd,%zd,%zd) converter\n", STRIDE, BLOCK_SIZE, BLOCK_STRIDE);
	    void *hybrid_soa_base = 0;
	    size_t hybrid_soa_stride = STRIDE;
	    size_t hybrid_soa_block_size = BLOCK_SIZE;
	    size_t hybrid_soa_block_stride = BLOCK_STRIDE;
	    bool ok = get_hybrid_soa_parameters(hybrid_soa_base, hybrid_soa_stride,
						hybrid_soa_block_size, hybrid_soa_block_stride);
	    assert(ok);
	    typename AT::template Typed<T, T> t(hybrid_soa_base, hybrid_soa_stride,
						hybrid_soa_block_size, hybrid_soa_block_stride);
            RegionAccessor<AT, T> result(t);
#ifdef POINTER_CHECKS
            result.set_impl(internal);
#endif
#ifdef PRIVILEGE_CHECKS
            result.set_privileges(priv);
#endif
            return result;
	  }

	  template <typename AT, typename REDOP>
	  bool can_convert_helper(ReductionFold<REDOP> *dummy) const {
	    void *redfold_base = 0;
	    bool ok = get_redfold_parameters(redfold_base);
	    return ok;
	  }

	  template <typename AT, typename REDOP>
	  RegionAccessor<AT, T> convert_helper(ReductionFold<REDOP> *dummy) const {
	    void *redfold_base = 0;
	    bool ok = get_redfold_parameters(redfold_base);
	    assert(ok);
	    typename AT::template Typed<T, T> t(redfold_base);
            RegionAccessor<AT, T> result(t);
#ifdef POINTER_CHECKS
            result.set_impl(internal);
#endif
#ifdef PRIVILEGE_CHECKS
            result.set_privileges(priv);
#endif
            return result;
	  }

	  template <typename AT, typename REDOP>
	  bool can_convert_helper(ReductionList<REDOP> *dummy) const {
	    void *redlist_base = 0;
	    ptr_t *redlist_next_ptr = 0;
	    bool ok = get_redlist_parameters(redlist_base, redlist_next_ptr);
	    return ok;
	  }

	  template <typename AT, typename REDOP>
	  RegionAccessor<AT, T> convert_helper(ReductionList<REDOP> *dummy) const {
	    void *redlist_base = 0;
	    ptr_t *redlist_next_ptr = 0;
	    bool ok = get_redlist_parameters(redlist_base, redlist_next_ptr);
	    assert(ok);
	    typename AT::template Typed<T, T> t(redlist_base, redlist_next_ptr);
            RegionAccessor<AT, T> result(t);
#ifdef POINTER_CHECKS
            result.set_impl(internal);
#endif
#ifdef PRIVILEGE_CHECKS
            result.set_privileges(priv);
#endif
            return result;
	  }
	};
      };

      template <typename T, typename PT> 
      struct Generic::StructSpecific<T, PT, true> {
	template <typename FT>
	RegionAccessor<Generic, FT, PT> get_field_accessor(FT T::* ptr) { 
	  Generic::Typed<T, PT> *rthis = static_cast<Generic::Typed<T, PT> *>(this);
	  return RegionAccessor<Generic, FT, PT>(Typed<FT, PT>(rthis->internal,
							       rthis->field_offset +
							       ((off_t)&(((T *)0)->*ptr))));
	}
      };

      template <size_t STRIDE> struct Stride : public Const<size_t, STRIDE> {
        CUDAPREFIX
        Stride(void) {}
        CUDAPREFIX
        Stride(size_t _value) : Const<size_t, STRIDE>(_value) {}
      };

      template <size_t BLOCK_SIZE> struct BlockSize : public Const<size_t, BLOCK_SIZE> {
        CUDAPREFIX
        BlockSize(size_t _value) : Const<size_t, BLOCK_SIZE>(_value) {}
      };

      template <size_t BLOCK_STRIDE> struct BlockStride : public Const<size_t, BLOCK_STRIDE> {
        CUDAPREFIX
        BlockStride(size_t _value) : Const<size_t, BLOCK_STRIDE>(_value) {}
      };

      template <size_t STRIDE> 
      struct AOS {
	struct Untyped : public Stride<STRIDE> {
          CUDAPREFIX
	  Untyped() : Stride<STRIDE>(), base(0) {}
          CUDAPREFIX
	  Untyped(void *_base, size_t _stride) : Stride<STRIDE>(_stride), base((char *)_base) {}
	  
          CUDAPREFIX
	  inline char *elem_ptr(ptr_t ptr) const
	  {
#ifdef POINTER_CHECKS
            verify_access(impl_ptr, ptr.value);
#endif
	    return(base + (ptr.value * Stride<STRIDE>::value));
	  }
	  //char *elem_ptr(const LowLevel::DomainPoint& dp) const;
	  //char *elem_ptr_linear(const LowLevel::Domain& d, LowLevel::Domain& subrect, ByteOffset *offsets);

	  char *base;
#ifdef POINTER_CHECKS
          void *impl_ptr;
#endif
#ifdef PRIVILEGE_CHECKS
        protected:
          AccessorPrivilege priv;
        public:
          inline void set_privileges_untyped(AccessorPrivilege p) { priv = p; }
#endif
	};

	template <typename T, typename PT>
	struct Typed : protected Untyped {
          CUDAPREFIX
	  Typed() : Untyped() {}
          CUDAPREFIX
	  Typed(void *_base, size_t _stride) : Untyped(_base, _stride) {}

#ifdef PRIVILEGE_CHECKS
          inline void set_privileges(AccessorPrivilege p) { this->priv = p; }
#endif

          CUDAPREFIX
	  inline T read(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_READ);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return *(const T *)(Untyped::elem_ptr(ptr)); 
          }
          CUDAPREFIX
	  inline void write(ptr_t ptr, const T& newval) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            *(T *)(Untyped::elem_ptr(ptr)) = newval; 
          }
          CUDAPREFIX
	  inline T *ptr(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return (T *)Untyped::elem_ptr(ptr); 
          }
          CUDAPREFIX
          inline T& ref(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return *((T*)Untyped::elem_ptr(ptr)); 
          }

	  template<typename REDOP> CUDAPREFIX
	  inline void reduce(ptr_t ptr, typename REDOP::RHS newval) const
	  {
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_REDUCE);
#endif
#ifdef POINTER_CHECKS
	    verify_access(this->template impl_ptr, ptr.value);
#endif
	    REDOP::template apply<false>(*(T *)Untyped::elem_ptr(ptr), newval);
	  }

#ifdef POINTER_CHECKS
          inline void set_impl(void *impl)
          {
            this->impl_ptr = impl;
          }
#endif

	  //T *elem_ptr(const LowLevel::DomainPoint& dp) const { return (T*)(Untyped::elem_ptr(dp)); }
	  //T *elem_ptr_linear(const LowLevel::Domain& d, LowLevel::Domain& subrect, ByteOffset *offsets)
	  //{ return (T*)(Untyped::elem_ptr_linear(d, subrect, offsets)); }
	};
      };

      template <size_t STRIDE> 
      struct SOA {
	struct Untyped : public Stride<STRIDE> {
          CUDAPREFIX
	  Untyped() : Stride<STRIDE>(0), base(0) {}
          CUDAPREFIX
	  Untyped(void *_base, size_t _stride) : Stride<STRIDE>(_stride), base((char *)_base) {}
	  
          CUDAPREFIX
	  inline char *elem_ptr(ptr_t ptr) const
	  {
	    return(base + (ptr.value * Stride<STRIDE>::value));
	  }

	  char *base;
#ifdef POINTER_CHECKS
          // TODO: Make pointer checks work on the GPU
          void *impl_ptr;
#endif
#ifdef PRIVILEGE_CHECKS
        protected:
          AccessorPrivilege priv;
        public:
          inline void set_privileges_untyped(AccessorPrivilege p) { priv = p; }
#endif
	};

	template <typename T, typename PT>
	struct Typed : protected Untyped {
          CUDAPREFIX
	  Typed() : Untyped() {}
          CUDAPREFIX
	  Typed(void *_base, size_t _stride) : Untyped(_base, _stride) {}

#ifdef PRIVILEGE_CHECKS
          inline void set_privileges(AccessorPrivilege p) { this->priv = p; }
#endif

          CUDAPREFIX
	  inline T read(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_READ);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return *(const T *)(Untyped::elem_ptr(ptr)); 
          }
          CUDAPREFIX
	  inline void write(ptr_t ptr, const T& newval) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            *(T *)(Untyped::elem_ptr(ptr)) = newval; 
          }
          CUDAPREFIX
	  inline T *ptr(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return (T *)Untyped::elem_ptr(ptr); 
          }
          CUDAPREFIX
          inline T& ref(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return *((T*)Untyped::elem_ptr(ptr)); 
          }

	  template<typename REDOP> CUDAPREFIX
	  inline void reduce(ptr_t ptr, typename REDOP::RHS newval) const
	  {
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_REDUCE);
#endif
#ifdef POINTER_CHECKS
	    verify_access(this->template impl_ptr, ptr.value);
#endif
	    REDOP::template apply<false>(*(T *)Untyped::elem_ptr(ptr), newval);
	  }

#ifdef POINTER_CHECKS
          inline void set_impl(void *impl)
          {
            this->impl_ptr = impl;
          }
#endif
	};
      };

      template <size_t STRIDE, size_t BLOCK_SIZE, size_t BLOCK_STRIDE> 
      struct HybridSOA {
	struct Untyped : public Stride<STRIDE>, public BlockSize<BLOCK_SIZE>, public BlockStride<BLOCK_STRIDE> {
          CUDAPREFIX
	  Untyped() : Stride<STRIDE>(0), BlockSize<BLOCK_SIZE>(0), BlockStride<BLOCK_STRIDE>(0), base(0) {}
          CUDAPREFIX
          Untyped(void *_base, size_t _stride, size_t _block_size, size_t _block_stride) 
	  : Stride<STRIDE>(_stride), BlockSize<BLOCK_SIZE>(_block_size),
	    BlockStride<BLOCK_STRIDE>(_block_stride), base((char *)_base) {}
	  
          CUDAPREFIX
	  inline char *elem_ptr(ptr_t ptr) const
	  {
#ifdef POINTER_CHECKS
            verify_access(impl_ptr, ptr.value);
#endif
	    return(base + (ptr.value * Stride<STRIDE>::value));
	  }

	  char *base;
#ifdef POINTER_CHECKS
          // TODO: Make this work for GPUs
          void *impl_ptr;
#endif
#ifdef PRIVILEGE_CHECKS
        protected:
          AccessorPrivilege priv;
        public:
          inline void set_privileges_untyped(AccessorPrivilege p) { priv = p; }
#endif
	};

	template <typename T, typename PT>
	struct Typed : protected Untyped {
          CUDAPREFIX
	  Typed() : Untyped() {}
          CUDAPREFIX
	  Typed(void *_base, size_t _stride, size_t _block_size, size_t _block_stride)
	    : Untyped(_base, _stride, _block_size, _block_stride) {}

#ifdef PRIVILEGE_CHECKS
          inline void set_privileges(AccessorPrivilege p) { this->priv = p; }
#endif

          CUDAPREFIX
	  inline T read(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_READ);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return *(const T *)(Untyped::elem_ptr(ptr)); 
          }
          CUDAPREFIX
	  inline void write(ptr_t ptr, const T& newval) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            *(T *)(Untyped::elem_ptr(ptr)) = newval; 
          }
          CUDAPREFIX
	  inline T *ptr(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return (T *)Untyped::elem_ptr(ptr); 
          }
          CUDAPREFIX
          inline T& ref(ptr_t ptr) const 
          { 
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_WRITE);
#endif
#ifdef POINTER_CHECKS
            verify_access(this->template impl_ptr, ptr.value);
#endif
            return *((T*)Untyped::elem_ptr(ptr)); 
          }

	  template<typename REDOP> CUDAPREFIX
	  inline void reduce(ptr_t ptr, typename REDOP::RHS newval) const
	  {
#ifdef PRIVILEGE_CHECKS
            assert(this->template priv & ACCESSOR_REDUCE);
#endif
#ifdef POINTER_CHECKS
	    verify_access(this->template impl_ptr, ptr.value);
#endif
	    REDOP::template apply<false>(*(T *)Untyped::elem_ptr(ptr), newval);
	  }

#ifdef POINTER_CHECKS
          inline void set_impl(void *impl)
          {
            this->impl_ptr = impl;
          }
#endif
	};
      };

      template <typename REDOP>
      struct ReductionFold {
	struct Untyped {
          CUDAPREFIX
	  Untyped() : base(0) {}
          CUDAPREFIX
	  Untyped(void *_base) : base((char *)_base) {}
	  
          CUDAPREFIX
	  inline char *elem_ptr(ptr_t ptr) const
	  {
#ifdef POINTER_CHECKS
            verify_access(impl_ptr, ptr.value);
#endif
	    return(base + (ptr.value * sizeof(typename REDOP::RHS)));
	  }

	  char *base;
#ifdef POINTER_CHECKS
          void *impl_ptr;
#endif
#ifdef PRIVILEGE_CHECKS
        protected:
          AccessorPrivilege priv;
        public:
          inline void set_privileges_untyped(AccessorPrivilege p) 
          { 
            assert((p == ACCESSOR_NONE) || (p == ACCESSOR_REDUCE));
            priv = p; 
          }
#endif
	};

	template <typename T, typename PT>
	struct Typed : protected Untyped {
          CUDAPREFIX
	  Typed(void) : Untyped() {}
          CUDAPREFIX
	  Typed(void *_base) : Untyped(_base) {}

#ifdef PRIVILEGE_CHECKS
          inline void set_privileges(AccessorPrivilege p) 
          { 
            assert((p == ACCESSOR_NONE) || (p == ACCESSOR_REDUCE));
            this->priv = p; 
          }
#endif

	  // only allowed operation on a reduction fold instance is a reduce (fold)
          CUDAPREFIX
	  inline void reduce(ptr_t ptr, typename REDOP::RHS newval) const
	  {
#ifdef PRIVILEGE_CHECKS
            assert(this->priv & ACCESSOR_REDUCE);
#endif
#ifdef POINTER_CHECKS
	    verify_access(this->impl_ptr, ptr.value);
#endif
	    REDOP::template fold<false>(*(typename REDOP::RHS *)Untyped::elem_ptr(ptr), newval);
	  }

#ifdef POINTER_CHECKS
          inline void set_impl(void *impl)
          {
            this->impl_ptr = impl;
          }
#endif
	};
      };

      template <typename REDOP>
      struct ReductionList {
	struct ReductionListEntry {
	  ptr_t ptr;
	  typename REDOP::RHS rhs;
	};

	struct Untyped {
	Untyped() : base(0), next_entry(0) {}
	Untyped(void *_base, ptr_t *_next_entry) : base((char *)_base), next_entry(_next_entry) {}
	  
	  inline char *elem_ptr(ptr_t ptr) const
	  {
#ifdef POINTER_CHECKS
            verify_access(impl_ptr, ptr.value);
#endif
	    return(base + (ptr.value * sizeof(ReductionListEntry)));
	  }

	  inline ptr_t get_next(void) const
	  {
	    ptr_t n;
#ifdef __GNUC__
	    n.value = __sync_fetch_and_add(&(next_entry->value), 1);
#else
            n.value = LowLevel::__sync_fetch_and_add(&(next_entry->value), 1); 
#endif
	    return n;
	  }

	  char *base;
	  ptr_t *next_entry;
#ifdef POINTER_CHECKS
          void *impl_ptr;
#endif
#ifdef PRIVILEGE_CHECKS
        protected:
          AccessorPrivilege priv;
        public:
          inline void set_untyped_privileges(AccessorPrivilege p) 
          { 
            assert((p == ACCESSOR_NONE) || (p == ACCESSOR_REDUCE));
            priv = p; 
          }
#endif
	};

	template <typename T, typename PT>
	struct Typed : protected Untyped {
	  Typed(void) : Untyped() {}
	  Typed(void *_base, ptr_t *_next_entry) : Untyped(_base, _next_entry) {}

#ifdef PRIVILEGE_CHECKS
          inline void set_privileges(AccessorPrivilege p) 
          { 
            assert((p == ACCESSOR_NONE) || (p == ACCESSOR_REDUCE));
            this->priv = p; 
          }
#endif

	  // only allowed operation on a reduction list instance is a reduce
	  inline void reduce(ptr_t ptr, typename REDOP::RHS newval) const
	  {
#ifdef PRIVILEGE_CHECKS
            assert(this->priv & ACCESSOR_REDUCE);
#endif
#ifdef POINTER_CHECKS
	    verify_access(this->impl_ptr, ptr.value);
#endif
	    ptr_t listptr = Untyped::get_next();
	    ReductionListEntry *entry = reinterpret_cast<ReductionListEntry *>(Untyped::elem_ptr(listptr));
	    entry->ptr = ptr;
	    entry->rhs = newval;
	  }

#ifdef POINTER_CHECKS
          inline void set_impl(void *impl)
          {
            this->impl_ptr = impl;
          }
#endif
	};
      };
    };

    template <typename AT, typename ET, typename PT> 
    struct RegionAccessor : public AT::template Typed<ET, PT> {
      CUDAPREFIX
      RegionAccessor()
	: AT::template Typed<ET, PT>() {}
      CUDAPREFIX
      RegionAccessor(const typename AT::template Typed<ET, PT>& to_copy) 
	: AT::template Typed<ET, PT>(to_copy) {}

      template <typename FT>
      struct FieldAccessor : 
        public AT::template Typed<ET, PT>::template Field<FT> {
        CUDAPREFIX
	FieldAccessor(void) {}

	//FieldAccessor(const typename AT::template Inner<ET, PT>::template Field<FT>& to_copy) {}
      };
    };
  };
};

#undef CUDAPREFIX
      
#endif
