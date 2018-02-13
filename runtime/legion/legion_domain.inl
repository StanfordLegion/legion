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

namespace Legion {

  // Specialization for 1-D Points
  template<typename T>
  struct Point<1,T> : public Realm::Point<1,T> {
  public:
    __CUDA_HD__
    inline Point(void) : Realm::Point<1,T>() { }
    __CUDA_HD__
    inline Point(const T val) : Realm::Point<1,T>(val) { }
    // copies allow type coercion (assuming the underlying type does)
    template<typename T2> __CUDA_HD__
    inline Point(const Point<1,T2> &rhs) : Realm::Point<1,T>(rhs) { }
    template<typename T2> __CUDA_HD__
    inline Point(const Realm::Point<1,T2> &rhs) : Realm::Point<1,T>(rhs) { }
  public:
    __CUDA_HD__
    inline operator T(void) const { return this->x; }
  public:
    __CUDA_HD__
    inline Point<1,T>& operator=(const T val)
      { this->x = val; return *this; }
    template<typename T2> __CUDA_HD__
    inline Point<1,T>& operator=(const Point<1,T2> &rhs)
      { this->x = rhs.x; return *this; }
    template<typename T2> __CUDA_HD__
    inline Point<1,T>& operator=(const Realm::Point<1,T2> &rhs)
      { this->x = rhs.x; return *this; }
  public:
    __CUDA_HD__
    inline static Point<1,T> ZEROES(void) { return Point<1,T>((T)0); }
    __CUDA_HD__
    inline static Point<1,T> ONES(void) { return Point<1,T>((T)1); }
  };

  // Specialization for 2-D Points
  template<typename T>
  struct Point<2,T> : public Realm::Point<2,T> {
  public:
    __CUDA_HD__
    inline Point(void) : Realm::Point<2,T>() { }
    __CUDA_HD__
    inline explicit Point(const T v) : Realm::Point<2,T>(v,v) { }
    __CUDA_HD__
    inline Point(const T v1, const T v2) : Realm::Point<2,T>(v1,v2) { }
    __CUDA_HD__
    inline explicit Point(const T vals[2]) : Realm::Point<2,T>(vals) { }
    // copies allow type coercion (assuming the underlying type does)
    template<typename T2> __CUDA_HD__
    inline Point(const Point<2,T2> &rhs) : Realm::Point<2,T>(rhs) { }
    template<typename T2> __CUDA_HD__
    inline Point(const Realm::Point<2,T2> &rhs) : Realm::Point<2,T>(rhs) { }
  public:
    template<typename T2> __CUDA_HD__
    inline Point<2,T>& operator=(const Point<2,T2> &rhs)
      { this->x = rhs.x; this->y = rhs.y; return *this; }
    template<typename T2> __CUDA_HD__
    inline Point<2,T>& operator=(const Realm::Point<2,T2> &rhs)
      { this->x = rhs.x; this->y = rhs.y; return *this; }
  public:
    __CUDA_HD__
    inline static Point<2,T> ZEROES(void) { return Point<2,T>((T)0); }
    __CUDA_HD__
    inline static Point<2,T> ONES(void) { return Point<2,T>((T)1); }
  };

  // Specialization for 3-D Points
  template<typename T>
  struct Point<3,T> : public Realm::Point<3,T> {
  public:
    __CUDA_HD__
    inline Point(void) : Realm::Point<3,T>() { }
    __CUDA_HD__
    inline explicit Point(const T v) : Realm::Point<3,T>(v,v,v) { }
    __CUDA_HD__
    inline Point(const T v1, const T v2, const T v3) 
      : Realm::Point<3,T>(v1,v2,v3) { }
    __CUDA_HD__
    inline explicit Point(const T vals[3]) : Realm::Point<3,T>(vals) { }
    // copies allow type coercion (assuming the underlying type does)
    template<typename T2> __CUDA_HD__
    inline Point(const Point<3,T2> &rhs) : Realm::Point<3,T>(rhs) { }
    template<typename T2> __CUDA_HD__
    inline Point(const Realm::Point<3,T2> &rhs) : Realm::Point<3,T>(rhs) { }
  public:
    template<typename T2> __CUDA_HD__
    inline Point<3,T>& operator=(const Point<3,T2> &rhs)
      { this->x = rhs.x; this->y = rhs.y; this->z = rhs.z; return *this; }
    template<typename T2> __CUDA_HD__
    inline Point<3,T>& operator=(const Realm::Point<3,T2> &rhs)
      { this->x = rhs.x; this->y = rhs.y; this->z = rhs.z; return *this; }
  public:
    __CUDA_HD__
    inline static Point<3,T> ZEROES(void) { return Point<3,T>((T)0); }
    __CUDA_HD__
    inline static Point<3,T> ONES(void) { return Point<3,T>((T)1); }
  };

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Point<DIM,T>::Point(void)
    : Realm::Point<DIM,T>()
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Point<DIM,T>::Point(const T val)
    : Realm::Point<DIM,T>()
  //----------------------------------------------------------------------------
  {
    this->x = val;
    this->y = val;
    this->z = val;
    this->w = val;
    for (int i = 4; i < DIM; i++)
      this->rest[i-4] = val;
  }
  
  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Point<DIM,T>::Point(const T vals[DIM])
    : Realm::Point<DIM,T>(vals)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2> __CUDA_HD__
  inline Point<DIM,T>::Point(const Point<DIM,T2> &rhs)
    : Realm::Point<DIM,T>(rhs)
  //----------------------------------------------------------------------------
  {
  }
  
  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2> __CUDA_HD__
  inline Point<DIM,T>::Point(const Realm::Point<DIM,T2> &rhs)
    : Realm::Point<DIM,T>(rhs)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2> __CUDA_HD__
  inline Point<DIM,T>& Point<DIM,T>::operator=(const Point<DIM,T2> &rhs)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < DIM; i++)
      (&Realm::Point<DIM,T>::x)[i] = (&rhs.x)[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2> __CUDA_HD__
  inline Point<DIM,T>& Point<DIM,T>::operator=(const Realm::Point<DIM,T2> &rhs)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < DIM; i++)
      (&Realm::Point<DIM,T>::x)[i] = (&rhs.x)[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  /*static*/ inline Point<DIM,T> Point<DIM,T>::ZEROES(void)
  //----------------------------------------------------------------------------
  {
    return Point<DIM,T>(0);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  /*static*/ inline Point<DIM,T> Point<DIM,T>::ONES(void)
  //----------------------------------------------------------------------------
  {
    return Point<DIM,T>(1);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Rect<DIM,T>::Rect(void)
    : Realm::Rect<DIM,T>()
  //----------------------------------------------------------------------------
  {
  }
  
  //----------------------------------------------------------------------------
  template<int DIM, typename T> __CUDA_HD__
  inline Rect<DIM,T>::Rect(const Point<DIM,T> &lo, const Point<DIM,T> &hi)
    : Realm::Rect<DIM,T>(lo,hi)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2> __CUDA_HD__
  inline Rect<DIM,T>::Rect(const Rect<DIM,T2> &rhs)
    : Realm::Rect<DIM,T>(rhs)
  //----------------------------------------------------------------------------
  {
  }
  
  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2> __CUDA_HD__
  inline Rect<DIM,T>::Rect(const Realm::Rect<DIM,T2> &rhs)
    : Realm::Rect<DIM,T>(rhs)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2> __CUDA_HD__
  inline Rect<DIM,T>& Rect<DIM,T>::operator=(const Rect<DIM,T2> &rhs)
  //----------------------------------------------------------------------------
  {
    this->lo = rhs.lo;
    this->hi = rhs.hi;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2> __CUDA_HD__
  inline Rect<DIM,T>& Rect<DIM,T>::operator=(const Realm::Rect<DIM,T2> &rhs)
  //----------------------------------------------------------------------------
  {
    this->lo = rhs.lo;
    this->hi = rhs.hi;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> __CUDA_HD__
  inline Transform<M,N,T>::Transform(void)
    : Realm::Matrix<M,N,T>()
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline Transform<M,N,T>::Transform(const Transform<M,N,T2> &rhs)
    : Realm::Matrix<M,N,T>(rhs)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline Transform<M,N,T>::Transform(const Realm::Matrix<M,N,T2> &rhs)
    : Realm::Matrix<M,N,T>(rhs)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline Transform<M,N,T>& Transform<M,N,T>::operator=(
                                                   const Transform<M,N,T2> &rhs)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < M; i++)
      this->rows[i] = rhs.rows[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline Transform<M,N,T>& Transform<M,N,T>::operator=(
                                              const Realm::Matrix<M,N,T2> &rhs)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < M; i++)
      this->rows[i] = rhs.rows[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> __CUDA_HD__
  inline AffineTransform<M,N,T>::AffineTransform(void)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        if (i == j)
          transform[i][j] = 1;
        else
          transform[i][j] = 0;
    for (int i = 0; i < M; i++)
      offset[i] = 0;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline AffineTransform<M,N,T>::AffineTransform(
                                             const AffineTransform<M,N,T2> &rhs)
    : transform(rhs.transform), offset(rhs.offset)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> 
    template<typename T2, typename T3> __CUDA_HD__
  inline AffineTransform<M,N,T>::AffineTransform(
                               const Transform<M,N,T2> t, const Point<M,T3> off)
    : transform(t), offset(off)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline AffineTransform<M,N,T>& AffineTransform<M,N,T>::operator=(
                                             const AffineTransform<M,N,T2> &rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    offset = rhs.offset;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline Point<M,T> AffineTransform<M,N,T>::operator[](
                                                  const Point<N,T2> point) const
  //----------------------------------------------------------------------------
  {
    return (transform * point) + offset;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> __CUDA_HD__
  inline bool AffineTransform<M,N,T>::is_identity(void) const
  //----------------------------------------------------------------------------
  {
    if (M != N)
      return false;
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        if (i == j) {
          if (transform[i][j] != 1)
            return false;
        } else {
          if (transform[i][j] != 0)
            return false;
        }
    for (int i = 0; i < M; i++)
      if (offset[i] != 0)
        return false;
    return true;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> __CUDA_HD__
  inline ScaleTransform<M,N,T>::ScaleTransform(void)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        if (i == j)
          transform[i][j] = 1;
        else
          transform[i][j] = 0;
    for (int i = 0; i < M; i++)
      extent.lo[i] = 0;
    extent.hi = extent.lo;
    for (int i = 0; i < M; i++)
      divisor[i] = 1;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline ScaleTransform<M,N,T>::ScaleTransform(
                                              const ScaleTransform<M,N,T2> &rhs)
    : transform(rhs.transform), extent(rhs.extent), divisor(rhs.divisor)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> 
    template<typename T2, typename T3, typename T4> __CUDA_HD__
  inline ScaleTransform<M,N,T>::ScaleTransform(const Transform<M,N,T2> t,
                                    const Rect<M,T3> ext, const Point<M,T4> div)
    : transform(t), extent(ext), divisor(div)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline ScaleTransform<M,N,T>& ScaleTransform<M,N,T>::operator=(
                                              const ScaleTransform<M,N,T2> &rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    extent = rhs.extent;
    divisor = rhs.divisor;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<typename T2> __CUDA_HD__
  inline Rect<M,T> ScaleTransform<M,N,T>::operator[](
                                                  const Point<N,T2> point) const
  //----------------------------------------------------------------------------
  {
    return ((transform * point) + extent) / divisor; 
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T> template<int P> __CUDA_HD__
  inline AffineTransform<M,P,T> AffineTransform<M,N,T>::operator()(
                                        const AffineTransform<N,P,T> &rhs) const
  //----------------------------------------------------------------------------
  {
    const Transform<M,P,T> t2 = transform * rhs.transform;
    const Point<M,T> p2 = transform * rhs.offset + offset;
    return AffineTransform<M,P,T>(t2, p2);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline bool ScaleTransform<M,N,T>::is_identity(void) const
  //----------------------------------------------------------------------------
  {
    if (M != N)
      return false;
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        if (i == j) {
          if (transform[i][j] != 1)
            return false;
        } else {
          if (transform[i][j] != 0)
            return false;
        }
    for (int i = 0; i < M; i++)
      if (extent.lo[i] != 0)
        return false;
    if (extent.lo != extent.hi)
      return false;
    for (int i = 0; i < M; i++)
      if (divisor[i] != 1)
        return false;
    return true;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline DomainT<DIM,T>::DomainT(void)
    : Realm::IndexSpace<DIM,T>()
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2>
  inline DomainT<DIM,T>::DomainT(const Rect<DIM,T2> &_bounds)
    : Realm::IndexSpace<DIM,T>(Realm::Rect<DIM,T>(_bounds))
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2>
  inline DomainT<DIM,T>::DomainT(const Realm::Rect<DIM,T2> &_bounds)
    : Realm::IndexSpace<DIM,T>(Realm::Rect<DIM,T>(_bounds))
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline DomainT<DIM,T>::DomainT(const DomainT<DIM,T> &rhs)
    : Realm::IndexSpace<DIM,T>(rhs)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline DomainT<DIM,T>::DomainT(const Realm::IndexSpace<DIM,T> &rhs)
    : Realm::IndexSpace<DIM,T>(rhs)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2>
  inline DomainT<DIM,T>& DomainT<DIM,T>::operator=(const Rect<DIM,T2> &rect)
  //----------------------------------------------------------------------------
  {
    this->bounds = rect;
    this->sparsity.id = 0;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T> template<typename T2>
  inline DomainT<DIM,T>& DomainT<DIM,T>::operator=(
                                               const Realm::Rect<DIM,T2> &rect)
  //----------------------------------------------------------------------------
  {
    this->bounds = rect;
    this->sparsity.id = 0;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline DomainT<DIM,T>& DomainT<DIM,T>::operator=(const DomainT<DIM,T> &rhs)
  //----------------------------------------------------------------------------
  {
    this->bounds = rhs.bounds;
    this->sparsity = rhs.sparsity;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline DomainT<DIM,T>& DomainT<DIM,T>::operator=(
                                           const Realm::IndexSpace<DIM,T> &rhs)
  //----------------------------------------------------------------------------
  {
    this->bounds = rhs.bounds;
    this->sparsity = rhs.sparsity;
    return *this;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint::DomainPoint(void)
    : dim(0)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < MAX_POINT_DIM; i++)
      point_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint::DomainPoint(coord_t index)
    : dim(1)
  //----------------------------------------------------------------------------
  {
    point_data[0] = index;
    for (int i = 1; i < MAX_POINT_DIM; i++)
      point_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint::DomainPoint(const DomainPoint &rhs)
    : dim(rhs.dim)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < MAX_POINT_DIM; i++)
      point_data[i] = rhs.point_data[i];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline DomainPoint::DomainPoint(const Point<DIM,T> &rhs)
    : dim(DIM)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < DIM; i++)
      point_data[i] = rhs[i];
    // Zero out the rest of the buffer to avoid uninitialized warnings
    for (int i = DIM; i < MAX_POINT_DIM; i++)
      point_data[i] = 0;
  }

  //----------------------------------------------------------------------------
  template<unsigned DIM>
  inline DomainPoint::operator LegionRuntime::Arrays::Point<DIM>(void) const
  //----------------------------------------------------------------------------
  {
    LegionRuntime::Arrays::Point<DIM> result;
    for (int i = 0; i < DIM; i++)
      result.x[i] = point_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline DomainPoint::operator Point<DIM,T>(void) const
  //----------------------------------------------------------------------------
  {
    assert(DIM == dim);
    Point<DIM,T> result;
    for (int i = 0; i < DIM; i++)
      result[i] = point_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint& DomainPoint::operator=(const DomainPoint &rhs)
  //----------------------------------------------------------------------------
  {
    dim = rhs.dim;
    for (int i = 0; i < dim; i++)
      point_data[i] = rhs.point_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  inline bool DomainPoint::operator==(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    if(dim != rhs.dim) return false;
    for(int i = 0; (i == 0) || (i < dim); i++)
      if(point_data[i] != rhs.point_data[i]) return false;
    return true;
  }

  //----------------------------------------------------------------------------
  inline bool DomainPoint::operator!=(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    return !((*this) == rhs);
  }

  //----------------------------------------------------------------------------
  inline bool DomainPoint::operator<(const DomainPoint &rhs) const
  //----------------------------------------------------------------------------
  {
    if (dim < rhs.dim) return true;
    if (dim > rhs.dim) return false;
    for (int i = 0; (i == 0) || (i < dim); i++) {
      if (point_data[i] < rhs.point_data[i]) return true;
      if (point_data[i] > rhs.point_data[i]) return false;
    }
    return false;
  }

  //----------------------------------------------------------------------------
  inline coord_t& DomainPoint::operator[](unsigned index)
  //----------------------------------------------------------------------------
  {
    assert(index < MAX_POINT_DIM);
    return point_data[index];
  }

  //----------------------------------------------------------------------------
  inline const coord_t& DomainPoint::operator[](unsigned index) const
  //----------------------------------------------------------------------------
  {
    assert(index < MAX_POINT_DIM);
    return point_data[index];
  }

  //----------------------------------------------------------------------------
  template<int DIM>
  /*static*/ inline DomainPoint 
                    DomainPoint::from_point(LegionRuntime::Arrays::Point<DIM> p)
  //----------------------------------------------------------------------------
  {
    DomainPoint dp;
    assert(DIM <= MAX_POINT_DIM);
    dp.dim = DIM;
    p.to_array(dp.point_data);
    return dp;
  }

  //----------------------------------------------------------------------------
  inline Color DomainPoint::get_color(void) const
  //----------------------------------------------------------------------------
  {
    assert(dim == 1);
    return point_data[0];
  }

  //----------------------------------------------------------------------------
  inline coord_t DomainPoint::get_index(void) const
  //----------------------------------------------------------------------------
  {
    assert(dim == 1);
    return point_data[0];
  }

  //----------------------------------------------------------------------------
  inline int DomainPoint::get_dim(void) const
  //----------------------------------------------------------------------------
  {
    return dim;
  }

  //----------------------------------------------------------------------------
  template<int DIM>
  inline LegionRuntime::Arrays::Point<DIM> DomainPoint::get_point(void) const
  //----------------------------------------------------------------------------
  {
    assert(dim == DIM); 
    return LegionRuntime::Arrays::Point<DIM>(point_data);
  }

  //----------------------------------------------------------------------------
  inline bool DomainPoint::is_null(void) const
  //----------------------------------------------------------------------------
  {
    return (dim == -1);
  }

  //----------------------------------------------------------------------------
  /*static*/ inline DomainPoint DomainPoint::nil(void)
  //----------------------------------------------------------------------------
  {
    DomainPoint p;
    p.dim = -1;
    return p;
  }

  //----------------------------------------------------------------------------
  inline /*friend */std::ostream& operator<<(std::ostream& os,
					     const DomainPoint& dp)
  //----------------------------------------------------------------------------
  {
    switch(dp.dim) {
    case 0: { os << '[' << dp.point_data[0] << ']'; break; }
    case 1: { os << '(' << dp.point_data[0] << ')'; break; }
    case 2: { os << '(' << dp.point_data[0]
                 << ',' << dp.point_data[1] << ')'; break; }
    case 3: { os << '(' << dp.point_data[0]
                 << ',' << dp.point_data[1]
                 << ',' << dp.point_data[2] << ')'; break; }
    default: assert(0);
    }
    return os;
  }

  //----------------------------------------------------------------------------
  inline Domain::Domain(void)
    : is_id(0), dim(0)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  inline Domain::Domain(const Domain &other)
    : is_id(other.is_id), dim(other.dim)
  //----------------------------------------------------------------------------
  {
    for(int i = 0; i < MAX_RECT_DIM*2; i++)
      rect_data[i] = other.rect_data[i];
  }

  //----------------------------------------------------------------------------
  inline Domain::Domain(const DomainPoint &lo, const DomainPoint &hi)
    : is_id(0), dim(lo.dim)
  //----------------------------------------------------------------------------
  {
    assert(lo.dim == hi.dim);
    for (int i = 0; i < dim; i++)
      rect_data[i] = lo[i];
    for (int i = 0; i < dim; i++)
      rect_data[i+dim] = hi[i];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline Domain::Domain(const Rect<DIM,T> &other)
    : is_id(0), dim(DIM)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < DIM; i++)
      rect_data[i] = other.lo[i];
    for (int i = 0; i < DIM; i++)
      rect_data[DIM+i] = other.hi[i];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline Domain::Domain(const DomainT<DIM,T> &other)
    : is_id(other.sparsity.id), dim(DIM)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < DIM; i++)
      rect_data[i] = other.bounds.lo[i];
    for (int i = 0; i < DIM; i++)
      rect_data[DIM+i] = other.bounds.hi[i];
  }

  //----------------------------------------------------------------------------
  inline Domain& Domain::operator=(const Domain &other)
  //----------------------------------------------------------------------------
  {
    is_id = other.is_id;
    dim = other.dim;
    for(int i = 0; i < MAX_RECT_DIM*2; i++)
      rect_data[i] = other.rect_data[i];
    return *this;
  }

  //----------------------------------------------------------------------------
  inline bool Domain::operator==(const Domain &rhs) const
  //----------------------------------------------------------------------------
  {
    if(is_id != rhs.is_id) return false;
    if(dim != rhs.dim) return false;
    for(int i = 0; i < dim; i++) {
      if(rect_data[i*2] != rhs.rect_data[i*2]) return false;
      if(rect_data[i*2+1] != rhs.rect_data[i*2+1]) return false;
    }
    return true;
  }

  //----------------------------------------------------------------------------
  inline bool Domain::operator!=(const Domain &rhs) const
  //----------------------------------------------------------------------------
  {
    return !(*this == rhs);
  }

  //----------------------------------------------------------------------------
  inline bool Domain::operator<(const Domain &rhs) const
  //----------------------------------------------------------------------------
  {
    if(is_id < rhs.is_id) return true;
    if(is_id > rhs.is_id) return false;
    if(dim < rhs.dim) return true;
    if(dim > rhs.dim) return false;
    for(int i = 0; i < 2*dim; i++) {
      if(rect_data[i] < rhs.rect_data[i]) return true;
      if(rect_data[i] > rhs.rect_data[i]) return false;
    }
    return false; // otherwise they are equal
  }

  //----------------------------------------------------------------------------
  inline bool Domain::exists(void) const
  //----------------------------------------------------------------------------
  {
    return (dim > 0);
  }

  //----------------------------------------------------------------------------
  inline bool Domain::dense(void) const
  //----------------------------------------------------------------------------
  {
    return (is_id == 0);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline Rect<DIM,T> Domain::bounds(void) const
  //----------------------------------------------------------------------------
  {
    assert(DIM == dim);
    Rect<DIM,T> result;
    for (int i = 0; i < DIM; i++)
      result.lo[i] = rect_data[i];
    for (int i = 0; i < DIM; i++)
      result.hi[i] = rect_data[DIM+i];
    return result;
  }

  //----------------------------------------------------------------------------
  template<int DIM>
  /*static*/ inline Domain Domain::from_rect(
                                    typename LegionRuntime::Arrays::Rect<DIM> r)
  //----------------------------------------------------------------------------
  {
    Domain d;
    assert(DIM <= MAX_RECT_DIM);
    d.dim = DIM;
    r.to_array(d.rect_data);
    return d;
  }

  //----------------------------------------------------------------------------
  template<int DIM>
  /*static*/ inline Domain Domain::from_point(
                                   typename LegionRuntime::Arrays::Point<DIM> p)
  //----------------------------------------------------------------------------
  {
    Domain d;
    assert(DIM <= MAX_RECT_DIM);
    d.dim = DIM;
    p.to_array(d.rect_data);
    p.to_array(d.rect_data+DIM);
    return d;
  }

  //----------------------------------------------------------------------------
  template<int DIM>
  inline Domain::operator LegionRuntime::Arrays::Rect<DIM>(void) const
  //----------------------------------------------------------------------------
  {
    assert(DIM == dim);
    assert(is_id == 0); // better not be one of these
    LegionRuntime::Arrays::Rect<DIM> result;
    for (int i = 0; i < DIM; i++)
      result.lo.x[i] = rect_data[i];
    for (int i = 0; i < DIM; i++)
      result.hi.x[i] = rect_data[DIM+i];
    return result;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline Domain::operator Rect<DIM,T>(void) const
  //----------------------------------------------------------------------------
  {
    assert(DIM == dim);
    assert(is_id == 0); // better not be one of these
    Rect<DIM,T> result;
    for (int i = 0; i < DIM; i++)
      result.lo[i] = rect_data[i];
    for (int i = 0; i < DIM; i++)
      result.hi[i] = rect_data[DIM+i];
    return result;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename T>
  inline Domain::operator DomainT<DIM,T>(void) const
  //----------------------------------------------------------------------------
  {
    assert(DIM == dim);
    DomainT<DIM,T> result;
    result.sparsity.id = is_id;
    for (int i = 0; i < DIM; i++)
      result.bounds.lo[i] = rect_data[i];
    for (int i = 0; i < DIM; i++)
      result.bounds.hi[i] = rect_data[DIM+i];
    return result;
  }

  //----------------------------------------------------------------------------
  /*static*/ inline Domain Domain::from_domain_point(const DomainPoint &p)
  //----------------------------------------------------------------------------
  {
    switch (p.dim) {
      case 0:
        assert(false);
      case 1:
        return Domain::from_point<1>(p.get_point<1>());
      case 2:
        return Domain::from_point<2>(p.get_point<2>());
      case 3:
        return Domain::from_point<3>(p.get_point<3>());
      default:
        assert(false);
    }
    return Domain::NO_DOMAIN;
  }

  //----------------------------------------------------------------------------
  inline bool Domain::is_valid(void) const
  //----------------------------------------------------------------------------
  {
    return exists();
  }

  //----------------------------------------------------------------------------
  inline bool Domain::contains(DomainPoint point) const
  //----------------------------------------------------------------------------
  {
    assert(point.get_dim() == dim);
    bool result = false;
    switch (dim)
    {
      case 1:
        {
          Point<1,coord_t> p1 = point;
          DomainT<1,coord_t> is1 = *this;
          result = is1.contains(p1);
          break;
        }
      case 2:
        {
          Point<2,coord_t> p2 = point;
          DomainT<2,coord_t> is2 = *this;
          result = is2.contains(p2);
          break;
        }
      case 3:
        {
          Point<3,coord_t> p3 = point;
          DomainT<3,coord_t> is3 = *this;
          result = is3.contains(p3);
          break;
        }
      default:
        assert(false);
    }
    return result;
  }

  //----------------------------------------------------------------------------
  inline int Domain::get_dim(void) const
  //----------------------------------------------------------------------------
  {
    return dim;
  }

  //----------------------------------------------------------------------------
  inline bool Domain::empty(void) const
  //----------------------------------------------------------------------------
  {
    return (get_volume() == 0);
  }

  //----------------------------------------------------------------------------
  inline size_t Domain::get_volume(void) const
  //----------------------------------------------------------------------------
  {
    switch (dim)
    {
      case 1:
        {
          DomainT<1,coord_t> is = *this;
          return is.volume();
        }
      case 2:
        {
          DomainT<2,coord_t> is = *this;
          return is.volume();
        }
      case 3:
        {
          DomainT<3,coord_t> is = *this;
          return is.volume();
        }
      default:
        assert(false);
    }
    return 0;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint Domain::lo(void) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = rect_data[i];
    return result;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint Domain::hi(void) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result;
    result.dim = dim;
    for (int i = 0; i < dim; i++)
      result[i] = rect_data[dim+i];
    return result;
  }

  //----------------------------------------------------------------------------
  inline Domain Domain::intersection(const Domain &other) const
  //----------------------------------------------------------------------------
  {
    assert(dim == other.dim);
    Realm::ProfilingRequestSet dummy_requests;
    switch (dim)
    {
      case 1:
        {
          DomainT<1,coord_t> is1 = *this;
          DomainT<1,coord_t> is2 = other;
          DomainT<1,coord_t> temp;
          Internal::LgEvent wait_on( 
            DomainT<1,coord_t>::compute_intersection(is1,is2,
                                                  temp,dummy_requests));
          if (wait_on.exists())
            wait_on.lg_wait();
          DomainT<1,coord_t> result = temp.tighten();
          temp.destroy();
          return Domain(result);
        }
      case 2:
        {
          DomainT<2,coord_t> is1 = *this;
          DomainT<2,coord_t> is2 = other;
          DomainT<2,coord_t> temp;
          Internal::LgEvent wait_on(
            DomainT<2,coord_t>::compute_intersection(is1,is2,
                                                  temp,dummy_requests));
          if (wait_on.exists())
            wait_on.lg_wait();
          DomainT<2,coord_t> result = temp.tighten();
          temp.destroy();
          return Domain(result);
        }
      case 3:
        {
          DomainT<3,coord_t> is1 = *this;
          DomainT<3,coord_t> is2 = other;
          DomainT<3,coord_t> temp;
          Internal::LgEvent wait_on(
            DomainT<3,coord_t>::compute_intersection(is1,is2,
                                                  temp,dummy_requests));
          if (wait_on.exists())
            wait_on.lg_wait();
          DomainT<3,coord_t> result = temp.tighten();
          temp.destroy();
          return Domain(result);
        }
      default:
        assert(false);
    }
    return Domain::NO_DOMAIN;
  }

  //----------------------------------------------------------------------------
  inline Domain Domain::convex_hull(const DomainPoint &p) const
  //----------------------------------------------------------------------------
  {
    assert(dim == p.dim);
    Realm::ProfilingRequestSet dummy_requests;
    switch (dim)
    {
      case 1:
        {
          Rect<1,coord_t> is1 = *this;
          Rect<1,coord_t> is2(p, p);
          Rect<1,coord_t> result = is1.union_bbox(is2);
          return Domain(result);
        }
      case 2:
        {
          Rect<2,coord_t> is1 = *this;
          Rect<2,coord_t> is2(p, p);
          Rect<2,coord_t> result = is1.union_bbox(is2);
          return Domain(result);
        }
      case 3:
        {
          Rect<3,coord_t> is1 = *this;
          Rect<3,coord_t> is2(p, p);
          Rect<3,coord_t> result = is1.union_bbox(is2);
          return Domain(result);
        }
      default:
        assert(false);
    }
    return Domain::NO_DOMAIN;
  }

  //----------------------------------------------------------------------------
  template<int DIM>
  inline LegionRuntime::Arrays::Rect<DIM> Domain::get_rect(void) const
  //----------------------------------------------------------------------------
  {
    assert(DIM > 0);
    assert(DIM == dim);
    // Runtime only returns tight domains so if it still has
    // a sparsity map then it is a real sparsity map
    assert(is_id == 0);
    return LegionRuntime::Arrays::Rect<DIM>(rect_data);
  }

  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator::DomainPointIterator(const Domain &d)
  //----------------------------------------------------------------------------
  {
    p.dim = d.get_dim();
    switch(p.get_dim()) {
    case 1:
      {
        Realm::IndexSpaceIterator<1,coord_t> *is_itr = 
          new (is_iterator) Realm::IndexSpaceIterator<1,coord_t>(
              DomainT<1,coord_t>(d));
        is_valid = is_itr->valid;
        if (is_valid) {
          Realm::PointInRectIterator<1,coord_t> *rect_itr = 
            new (rect_iterator) 
              Realm::PointInRectIterator<1,coord_t>(is_itr->rect);
          rect_valid = rect_itr->valid;
          p = Point<1,coord_t>(rect_itr->p); 
        } else {
          rect_valid = false;
        }
        break;
      }
    case 2:
      {
        Realm::IndexSpaceIterator<2,coord_t> *is_itr = 
          new (is_iterator) Realm::IndexSpaceIterator<2,coord_t>(
              DomainT<2,coord_t>(d));
        is_valid = is_itr->valid;
        if (is_valid) {
          Realm::PointInRectIterator<2,coord_t> *rect_itr = 
            new (rect_iterator)
              Realm::PointInRectIterator<2,coord_t>(is_itr->rect);
          rect_valid = rect_itr->valid;
          p = Point<2,coord_t>(rect_itr->p); 
        } else {
          rect_valid = false;
        }
        break;
      }
    case 3:
      {
        Realm::IndexSpaceIterator<3,coord_t> *is_itr = 
          new (is_iterator) Realm::IndexSpaceIterator<3,coord_t>(
              DomainT<3,coord_t>(d));
        is_valid = is_itr->valid;
        if (is_valid) {
          Realm::PointInRectIterator<3,coord_t> *rect_itr = 
            new (rect_iterator) 
              Realm::PointInRectIterator<3,coord_t>(is_itr->rect);
          rect_valid = rect_itr->valid;
          p = Point<3,coord_t>(rect_itr->p); 
        } else {
          rect_valid = false;
        }
        break;
      }
    default:
      assert(0);
    };
  }

  //----------------------------------------------------------------------------
  inline bool Domain::DomainPointIterator::step(void)
  //----------------------------------------------------------------------------
  {
    assert(is_valid && rect_valid);
    switch(p.get_dim()) {
    case 1:
      {
        // Step the rect iterator first
        Realm::PointInRectIterator<1,coord_t> *rect_itr = 
          (Realm::PointInRectIterator<1,coord_t>*)(void *)rect_iterator;
        rect_itr->step();
        rect_valid = rect_itr->valid;
        if (!rect_valid) {
          // If the rectangle iterator is not valid anymore
          // then try to start the next rectangle
          Realm::IndexSpaceIterator<1,coord_t> *is_itr = 
            (Realm::IndexSpaceIterator<1,coord_t>*)(void *)is_iterator;
          is_itr->step();
          is_valid = is_itr->valid;
          if (is_valid) {
            // Placement new on top of the old one
            new (rect_itr) 
              Realm::PointInRectIterator<1,coord_t>(is_itr->rect);
            p = Point<1,coord_t>(rect_itr->p);
            rect_valid = rect_itr->valid;
          } else {
            rect_valid = false;
          }
        } else {
          p = Point<1,coord_t>(rect_itr->p); 
        }
        break;
      }
    case 2:
      {
        // Step the rect iterator first
        Realm::PointInRectIterator<2,coord_t> *rect_itr = 
          (Realm::PointInRectIterator<2,coord_t>*)(void *)rect_iterator;
        rect_itr->step();
        rect_valid = rect_itr->valid;
        if (!rect_valid) {
          // If the rectangle iterator is not valid anymore
          // then try to start the next rectangle
          Realm::IndexSpaceIterator<2,coord_t> *is_itr = 
            (Realm::IndexSpaceIterator<2,coord_t>*)(void *)is_iterator;
          is_itr->step();
          is_valid = is_itr->valid;
          if (is_valid) {
            // Placement new on top of the old one
            new (rect_itr) 
              Realm::PointInRectIterator<2,coord_t>(is_itr->rect);
            p = Point<2,coord_t>(rect_itr->p);
            rect_valid = rect_itr->valid;
          } else {
            rect_valid = false;
          }
        } else {
          p = Point<2,coord_t>(rect_itr->p); 
        }
        break;
      }
    case 3:
      {
        // Step the rect iterator first
        Realm::PointInRectIterator<3,coord_t> *rect_itr = 
          (Realm::PointInRectIterator<3,coord_t>*)(void *)rect_iterator;
        rect_itr->step();
        rect_valid = rect_itr->valid;
        if (!rect_valid) {
          // If the rectangle iterator is not valid anymore
          // then try to start the next rectangle
          Realm::IndexSpaceIterator<3,coord_t> *is_itr = 
            (Realm::IndexSpaceIterator<3,coord_t>*)(void *)is_iterator;
          is_itr->step();
          is_valid = is_itr->valid;
          if (is_valid) {
            // Placement new on top of the old one
            new (rect_itr) 
              Realm::PointInRectIterator<3,coord_t>(is_itr->rect);
            p = Point<3,coord_t>(rect_itr->p);
            rect_valid = rect_itr->valid;
          } else {
            rect_valid = false;
          }
        } else {
          p = Point<3,coord_t>(rect_itr->p); 
        }
        break;
      }
    default:
      assert(0);
    }
    return is_valid && rect_valid;
  }

  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator::operator bool(void) const
  //----------------------------------------------------------------------------
  {
    return is_valid && rect_valid;
  }

  //----------------------------------------------------------------------------
  inline Domain::DomainPointIterator& Domain::DomainPointIterator::operator++(
                                                                 int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  inline std::ostream& operator<<(std::ostream &os, Domain d)
  //----------------------------------------------------------------------------
  {
    switch(d.get_dim()) {
    case 1: return os << d.get_rect<1>();
    case 2: return os << d.get_rect<2>();
    case 3: return os << d.get_rect<3>();
    default: assert(0);
    }
    return os;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInRectIterator<DIM,COORD_T>::PointInRectIterator(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInRectIterator<DIM,COORD_T>::PointInRectIterator(
             const Rect<DIM,COORD_T> &r, bool column_major_order)
    : itr(Realm::PointInRectIterator<DIM,COORD_T>(r, column_major_order))
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInRectIterator<DIM,COORD_T>::valid(void) const
  //----------------------------------------------------------------------------
  {
    return itr.valid;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInRectIterator<DIM,COORD_T>::step(void)
  //----------------------------------------------------------------------------
  {
    assert(valid());
    itr.step();
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInRectIterator<DIM,COORD_T>::operator()(void) const
  //----------------------------------------------------------------------------
  {
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline const Point<DIM,COORD_T>& 
                         PointInRectIterator<DIM,COORD_T>::operator*(void) const
  //----------------------------------------------------------------------------
  {
    current = itr.p;
    return current;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline COORD_T 
              PointInRectIterator<DIM,COORD_T>::operator[](unsigned index) const
  //----------------------------------------------------------------------------
  {
    return itr.p[index];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline const Point<DIM,COORD_T>* 
                        PointInRectIterator<DIM,COORD_T>::operator->(void) const
  //----------------------------------------------------------------------------
  {
    current = itr.p;
    return &current;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInRectIterator<DIM,COORD_T>&
                              PointInRectIterator<DIM,COORD_T>::operator++(void)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInRectIterator<DIM,COORD_T>&
                    PointInRectIterator<DIM,COORD_T>::operator++(int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline RectInDomainIterator<DIM,COORD_T>::RectInDomainIterator(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline RectInDomainIterator<DIM,COORD_T>::RectInDomainIterator(
                                       const DomainT<DIM,COORD_T> &d)
    : itr(Realm::IndexSpaceIterator<DIM,COORD_T>(d))
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool RectInDomainIterator<DIM,COORD_T>::valid(void) const
  //----------------------------------------------------------------------------
  {
    return itr.valid;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool RectInDomainIterator<DIM,COORD_T>::step(void)
  //----------------------------------------------------------------------------
  {
    assert(valid());
    itr.step();
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool RectInDomainIterator<DIM,COORD_T>::operator()(void) const
  //----------------------------------------------------------------------------
  {
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline const Rect<DIM,COORD_T>&
                        RectInDomainIterator<DIM,COORD_T>::operator*(void) const
  //----------------------------------------------------------------------------
  {
    current = itr.rect;
    return current;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline const Rect<DIM,COORD_T>*
                       RectInDomainIterator<DIM,COORD_T>::operator->(void) const
  //----------------------------------------------------------------------------
  {
    current = itr.rect;
    return current;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline RectInDomainIterator<DIM,COORD_T>& 
                             RectInDomainIterator<DIM,COORD_T>::operator++(void)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline RectInDomainIterator<DIM,COORD_T>&
                   RectInDomainIterator<DIM,COORD_T>::operator++(int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInDomainIterator<DIM,COORD_T>::PointInDomainIterator(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInDomainIterator<DIM,COORD_T>::PointInDomainIterator(
              const DomainT<DIM,COORD_T> &d, bool column_major_order)
    : rect_itr(RectInDomainIterator<DIM,COORD_T>(d)),
      column_major(column_major_order)
  //----------------------------------------------------------------------------
  {
    if (rect_itr())
      point_itr = PointInRectIterator<DIM,COORD_T>(*rect_itr, column_major);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInDomainIterator<DIM,COORD_T>::valid(void) const
  //----------------------------------------------------------------------------
  {
    return point_itr();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInDomainIterator<DIM,COORD_T>::step(void)
  //----------------------------------------------------------------------------
  {
    assert(valid()); 
    point_itr++;
    if (!point_itr())
    {
      rect_itr++;
      if (rect_itr())
        point_itr = PointInRectIterator<DIM,COORD_T>(*rect_itr, column_major);
    }
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline bool PointInDomainIterator<DIM,COORD_T>::operator()(void) const
  //----------------------------------------------------------------------------
  {
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline const Point<DIM,COORD_T>& 
                       PointInDomainIterator<DIM,COORD_T>::operator*(void) const
  //----------------------------------------------------------------------------
  {
    current = *point_itr;
    return current;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline const Point<DIM,COORD_T>* 
                      PointInDomainIterator<DIM,COORD_T>::operator->(void) const
  //----------------------------------------------------------------------------
  {
    current = *point_itr;
    return &current;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline COORD_T 
            PointInDomainIterator<DIM,COORD_T>::operator[](unsigned index) const
  //----------------------------------------------------------------------------
  {
    return point_itr[index];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInDomainIterator<DIM,COORD_T>&
                            PointInDomainIterator<DIM,COORD_T>::operator++(void)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  inline PointInDomainIterator<DIM,COORD_T>&
                  PointInDomainIterator<DIM,COORD_T>::operator++(int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  inline DomainTransform::DomainTransform(void)
    : m(0), n(0)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  inline DomainTransform::DomainTransform(const DomainTransform &rhs)
    : m(rhs.m), n(rhs.n)
  //----------------------------------------------------------------------------
  {
    assert(m < ::MAX_POINT_DIM);
    assert(n < ::MAX_POINT_DIM);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        matrix[i * n + j] = rhs.matrix[i * n + j];
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainTransform::DomainTransform(const Transform<M,N,T> &rhs)
    : m(M), n(N)
  //----------------------------------------------------------------------------
  {
    assert(m < ::MAX_POINT_DIM);
    assert(n < ::MAX_POINT_DIM);
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        matrix[i * n + j] = rhs[i][j];
  }

  //----------------------------------------------------------------------------
  inline DomainTransform& DomainTransform::operator=(const DomainTransform &rhs)
  //----------------------------------------------------------------------------
  {
    m = rhs.m;
    n = rhs.n;
    assert(m < ::MAX_POINT_DIM);
    assert(n < ::MAX_POINT_DIM);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        matrix[i * n + j] = rhs.matrix[i * n + j];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainTransform& 
                         DomainTransform::operator=(const Transform<M,N,T> &rhs)
  //----------------------------------------------------------------------------
  {
    m = M;
    n = N;
    assert(m < ::MAX_POINT_DIM);
    assert(n < ::MAX_POINT_DIM);
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        matrix[i * n + j] = rhs[i][j];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainTransform::operator Transform<M,N,T>(void) const
  //----------------------------------------------------------------------------
  {
    assert(M == m);
    assert(N == n);
    Transform<M,N,T> result;
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        result[i][j] = matrix[i * n + j];
    return result;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint DomainTransform::operator*(const DomainPoint &p) const
  //----------------------------------------------------------------------------
  {
    assert(n == p.dim);
    DomainPoint result;
    result.dim = m;
    for (int i = 0; i < m; i++)
    {
      result.point_data[i] = 0;
      for (int j = 0; j < n; j++)
        result.point_data[i] += matrix[i * n + j] * p.point_data[j];
    }
    return result;
  }

  //----------------------------------------------------------------------------
  inline bool DomainTransform::is_identity(void) const
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        if (i == j)
        {
          if (matrix[i * n + j] != 1)
            return false;
        }
        else
        {
          if (matrix[i * n + j] != 0)
            return false;
        }
    return true;
  }

  //----------------------------------------------------------------------------
  inline DomainAffineTransform::DomainAffineTransform(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  inline DomainAffineTransform::DomainAffineTransform(
                                               const DomainAffineTransform &rhs)
    : transform(rhs.transform), offset(rhs.offset)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == offset.dim);
  }

  //----------------------------------------------------------------------------
  inline DomainAffineTransform::DomainAffineTransform(
                                 const DomainTransform &t, const DomainPoint &p)
    : transform(t), offset(p)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == offset.dim);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainAffineTransform::DomainAffineTransform(
                                              const AffineTransform<M,N,T> &rhs)
    : transform(rhs.transform), offset(rhs.offset)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == offset.dim);
  }

  //----------------------------------------------------------------------------
  inline DomainAffineTransform& DomainAffineTransform::operator=(
                                               const DomainAffineTransform &rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    offset = rhs.offset;
    assert(transform.m == offset.dim);
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainAffineTransform& DomainAffineTransform::operator=(
                                              const AffineTransform<M,N,T> &rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    offset = rhs.offset;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainAffineTransform::operator AffineTransform<M,N,T>(void) const
  //----------------------------------------------------------------------------
  {
    AffineTransform<M,N,T> result;
    result.transform = transform;
    result.offset = offset;
    return result;
  }

  //----------------------------------------------------------------------------
  inline DomainPoint DomainAffineTransform::operator[](
                                                     const DomainPoint &p) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result = transform * p;
    for (int i = 0; i < result.dim; i++)
      result[i] += offset[i];
    return result;
  }

  //----------------------------------------------------------------------------
  inline bool DomainAffineTransform::is_identity(void) const
  //----------------------------------------------------------------------------
  {
    if (!transform.is_identity())
      return false;
    for (int i = 0; i < offset.dim; i++)
      if (offset.point_data[i] != 0)
        return false;
    return true;
  }

  //----------------------------------------------------------------------------
  inline DomainScaleTransform::DomainScaleTransform(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  inline DomainScaleTransform::DomainScaleTransform(
                                                const DomainScaleTransform &rhs)
    : transform(rhs.transform), extent(rhs.extent), divisor(rhs.divisor)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == divisor.dim);
    assert(transform.m == extent.dim);
  }

  //----------------------------------------------------------------------------
  inline DomainScaleTransform::DomainScaleTransform(const DomainTransform &t,
                                          const Domain &e, const DomainPoint &d)
    : transform(t), extent(e), divisor(d)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == divisor.dim);
    assert(transform.m == extent.dim);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainScaleTransform::DomainScaleTransform(
                                               const ScaleTransform<M,N,T> &rhs)
    : transform(rhs.transform), extent(rhs.extent), divisor(rhs.divisor)
  //----------------------------------------------------------------------------
  {
  }
  
  //----------------------------------------------------------------------------
  inline DomainScaleTransform& DomainScaleTransform::operator=(
                                                const DomainScaleTransform &rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    extent = rhs.extent;
    divisor = rhs.divisor;
    assert(transform.m == divisor.dim);
    assert(transform.m == extent.dim);
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainScaleTransform& DomainScaleTransform::operator=(
                                               const ScaleTransform<M,N,T> &rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    extent = rhs.extent;
    divisor = rhs.divisor;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  inline DomainScaleTransform::operator ScaleTransform<M,N,T>(void) const
  //----------------------------------------------------------------------------
  {
    ScaleTransform<M,N,T> result;
    result.transform = transform;
    result.extent = extent;
    result.divisor = divisor;
    return result;
  }

  //----------------------------------------------------------------------------
  inline Domain DomainScaleTransform::operator[](const DomainPoint &p) const
  //----------------------------------------------------------------------------
  {
    DomainPoint p2 = transform * p;
    DomainPoint lo, hi;
    for (int i = 0; i < p2.dim; i++)
      lo[i] = (extent.lo()[i] + p2[i]) / divisor[i];
    for (int i = 0; i < p2.dim; i++)
      hi[i] = (extent.hi()[i] + p2[i]) / divisor[i];
    return Domain(lo, hi);
  }

  //----------------------------------------------------------------------------
  inline bool DomainScaleTransform::is_identity(void) const
  //----------------------------------------------------------------------------
  {
    if (!transform.is_identity())
      return false;
    if (extent.lo() != extent.hi())
      return false;
    for (int i = 0; i < divisor.dim; i++)
      if (divisor[i] != 1)
        return false;
    return true;
  }

}; // namespace Legion

// Specializations of std::less<T> for Point,Rect,DomainT for use in containers
namespace std {
  template<int DIM, typename T>
  struct less<Legion::Point<DIM,T> > {
    inline bool operator()(const Legion::Point<DIM,T>& p1, 
                           const Legion::Point<DIM,T>& p2) const
    { return std::less<Realm::Point<DIM,T> >()(p1, p2); }
  };

  template<int DIM, typename T>
  struct less<Legion::Rect<DIM,T> > {
    inline bool operator()(const Legion::Rect<DIM,T>& r1, 
                           const Legion::Rect<DIM,T>& r2) const
    { return std::less<Realm::Rect<DIM,T> >()(r1, r2); }
  };

  template<int DIM, typename T>
  struct less<Legion::DomainT<DIM,T> > {
    inline bool operator()(const Legion::DomainT<DIM,T>& d1, 
                           const Legion::DomainT<DIM,T>& d2) const
    { return std::less<Realm::IndexSpace<DIM,T> >()(d1, d2); }
  };
};

