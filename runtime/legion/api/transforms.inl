/* Copyright 2026 Stanford University, NVIDIA Corporation
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

// Included from transforms.h - do not include this directly

// Useful for IDEs
#include "legion/api/transforms.h"

namespace Legion {

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline AffineTransform<M, N, T>::AffineTransform(void)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        if (i == j)
          transform[i][j] = 1;
        else
          transform[i][j] = 0;
    for (int i = 0; i < M; i++) offset[i] = 0;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<typename T2>
  __LEGION_CUDA_HD__ inline AffineTransform<M, N, T>::AffineTransform(
      const AffineTransform<M, N, T2>& rhs)
    : transform(rhs.transform), offset(rhs.offset)
  //----------------------------------------------------------------------------
  { }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<typename T2, typename T3>
  __LEGION_CUDA_HD__ inline AffineTransform<M, N, T>::AffineTransform(
      const Transform<M, N, T2> t, const Point<M, T3> off)
    : transform(t), offset(off)
  //----------------------------------------------------------------------------
  { }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<typename T2>
  __LEGION_CUDA_HD__ inline AffineTransform<M, N, T>&
      AffineTransform<M, N, T>::operator=(const AffineTransform<M, N, T2>& rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    offset = rhs.offset;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<typename T2>
  __LEGION_CUDA_HD__ inline Point<M, T> AffineTransform<M, N, T>::operator[](
      const Point<N, T2> point) const
  //----------------------------------------------------------------------------
  {
    return (transform * point) + offset;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline bool AffineTransform<M, N, T>::is_identity(
      void) const
  //----------------------------------------------------------------------------
  {
    if (M == N)
    {
      for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
          if (i == j)
          {
            if (transform[i][j] != 1)
              return false;
          }
          else
          {
            if (transform[i][j] != 0)
              return false;
          }
      for (int i = 0; i < M; i++)
        if (offset[i] != 0)
          return false;
      return true;
    }
    else
      return false;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline ScaleTransform<M, N, T>::ScaleTransform(void)
  //----------------------------------------------------------------------------
  {
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        if (i == j)
          transform[i][j] = 1;
        else
          transform[i][j] = 0;
    for (int i = 0; i < M; i++) extent.lo[i] = 0;
    extent.hi = extent.lo;
    for (int i = 0; i < M; i++) divisor[i] = 1;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<typename T2>
  __LEGION_CUDA_HD__ inline ScaleTransform<M, N, T>::ScaleTransform(
      const ScaleTransform<M, N, T2>& rhs)
    : transform(rhs.transform), extent(rhs.extent), divisor(rhs.divisor)
  //----------------------------------------------------------------------------
  { }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<typename T2, typename T3, typename T4>
  __LEGION_CUDA_HD__ inline ScaleTransform<M, N, T>::ScaleTransform(
      const Transform<M, N, T2> t, const Rect<M, T3> ext,
      const Point<M, T4> div)
    : transform(t), extent(ext), divisor(div)
  //----------------------------------------------------------------------------
  { }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<typename T2>
  __LEGION_CUDA_HD__ inline ScaleTransform<M, N, T>&
      ScaleTransform<M, N, T>::operator=(const ScaleTransform<M, N, T2>& rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    extent = rhs.extent;
    divisor = rhs.divisor;
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<typename T2>
  __LEGION_CUDA_HD__ inline Rect<M, T> ScaleTransform<M, N, T>::operator[](
      const Point<N, T2> point) const
  //----------------------------------------------------------------------------
  {
    return ((transform * point) + extent) / divisor;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  template<int P>
  __LEGION_CUDA_HD__ inline AffineTransform<M, P, T>
      AffineTransform<M, N, T>::operator()(
          const AffineTransform<N, P, T>& rhs) const
  //----------------------------------------------------------------------------
  {
    const Transform<M, P, T> t2 = transform * rhs.transform;
    const Point<M, T> p2 = transform * rhs.offset + offset;
    return AffineTransform<M, P, T>(t2, p2);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline bool ScaleTransform<M, N, T>::is_identity(
      void) const
  //----------------------------------------------------------------------------
  {
    if (M == N)
    {
      for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
          if (i == j)
          {
            if (transform[i][j] != 1)
              return false;
          }
          else
          {
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
    else
      return false;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline DomainTransform::DomainTransform(void) : m(0), n(0)
  //----------------------------------------------------------------------------
  { }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline DomainTransform::DomainTransform(const DomainTransform& rhs)
    : m(rhs.m), n(rhs.n)
  //----------------------------------------------------------------------------
  {
    assert(m <= LEGION_MAX_DIM);
    assert(n <= LEGION_MAX_DIM);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) matrix[i * n + j] = rhs.matrix[i * n + j];
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline DomainTransform::DomainTransform(
      const Transform<M, N, T>& rhs)
    : m(M), n(N)
  //----------------------------------------------------------------------------
  {
    assert(m <= LEGION_MAX_DIM);
    assert(n <= LEGION_MAX_DIM);
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++) matrix[i * n + j] = rhs[i][j];
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline DomainTransform& DomainTransform::operator=(const DomainTransform& rhs)
  //----------------------------------------------------------------------------
  {
    m = rhs.m;
    n = rhs.n;
    assert(m <= LEGION_MAX_DIM);
    assert(n <= LEGION_MAX_DIM);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) matrix[i * n + j] = rhs.matrix[i * n + j];
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline DomainTransform& DomainTransform::operator=(
      const Transform<M, N, T>& rhs)
  //----------------------------------------------------------------------------
  {
    m = M;
    n = N;
    assert(m <= LEGION_MAX_DIM);
    assert(n <= LEGION_MAX_DIM);
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++) matrix[i * n + j] = rhs[i][j];
    return *this;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline bool DomainTransform::operator==(const DomainTransform& rhs) const
  //----------------------------------------------------------------------------
  {
    if (m != rhs.m)
      return false;
    if (n != rhs.n)
      return false;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        if (matrix[i * n + j] != rhs.matrix[i * n + j])
          return false;
    return true;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline bool DomainTransform::operator!=(const DomainTransform& rhs) const
  //----------------------------------------------------------------------------
  {
    return !(*this == rhs);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline DomainTransform::operator Transform<M, N, T>(
      void) const
  //----------------------------------------------------------------------------
  {
    assert(M == m);
    assert(N == n);
    Transform<M, N, T> result;
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++) result[i][j] = matrix[i * n + j];
    return result;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline DomainPoint DomainTransform::operator*(const DomainPoint& p) const
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
  __LEGION_CUDA_HD__
  inline Domain DomainTransform::operator*(const Domain& domain) const
  //----------------------------------------------------------------------------
  {
    assert(domain.dense());
    assert(n == domain.get_dim());
    DomainPoint lo = this->operator*(domain.lo());
    DomainPoint hi = this->operator*(domain.hi());
    return Domain(lo, hi);
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline DomainTransform DomainTransform::operator*(
      const DomainTransform& rhs) const
  //----------------------------------------------------------------------------
  {
    assert(n == rhs.m);
    DomainTransform result;
    result.m = m;
    result.n = rhs.n;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < rhs.n; j++)
      {
        coord_t product = 0;
        for (int k = 0; k < n; k++)
          product += (matrix[i * n + k] * rhs.matrix[k * rhs.n + j]);
        result.matrix[i * rhs.n + j] = product;
      }
    return result;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline bool DomainTransform::is_identity(void) const
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
  __LEGION_CUDA_HD__ inline DomainAffineTransform::DomainAffineTransform(void)
  //----------------------------------------------------------------------------
  { }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline DomainAffineTransform::DomainAffineTransform(
      const DomainAffineTransform& rhs)
    : transform(rhs.transform), offset(rhs.offset)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == offset.dim);
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline DomainAffineTransform::DomainAffineTransform(
      const DomainTransform& t, const DomainPoint& p)
    : transform(t), offset(p)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == offset.dim);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline DomainAffineTransform::DomainAffineTransform(
      const AffineTransform<M, N, T>& rhs)
    : transform(rhs.transform), offset(rhs.offset)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == offset.dim);
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline DomainAffineTransform&
      DomainAffineTransform::operator=(const DomainAffineTransform& rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    offset = rhs.offset;
    assert(transform.m == offset.dim);
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline DomainAffineTransform&
      DomainAffineTransform::operator=(const AffineTransform<M, N, T>& rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    offset = rhs.offset;
    return *this;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline bool DomainAffineTransform::operator==(
      const DomainAffineTransform& rhs) const
  //----------------------------------------------------------------------------
  {
    if (transform != rhs.transform)
      return false;
    if (offset != rhs.offset)
      return false;
    return true;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline bool DomainAffineTransform::operator!=(
      const DomainAffineTransform& rhs) const
  //----------------------------------------------------------------------------
  {
    return !(*this == rhs);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline DomainAffineTransform::operator AffineTransform<
      M, N, T>(void) const
  //----------------------------------------------------------------------------
  {
    AffineTransform<M, N, T> result;
    result.transform = transform;
    result.offset = offset;
    return result;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline DomainPoint DomainAffineTransform::operator[](
      const DomainPoint& p) const
  //----------------------------------------------------------------------------
  {
    DomainPoint result = transform * p;
    for (int i = 0; i < result.dim; i++) result[i] += offset[i];
    return result;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline bool DomainAffineTransform::is_identity(void) const
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
  __LEGION_CUDA_HD__ inline DomainScaleTransform::DomainScaleTransform(void)
  //----------------------------------------------------------------------------
  { }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline DomainScaleTransform::DomainScaleTransform(
      const DomainScaleTransform& rhs)
    : transform(rhs.transform), extent(rhs.extent), divisor(rhs.divisor)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == divisor.dim);
    assert(transform.m == extent.dim);
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline DomainScaleTransform::DomainScaleTransform(
      const DomainTransform& t, const Domain& e, const DomainPoint& d)
    : transform(t), extent(e), divisor(d)
  //----------------------------------------------------------------------------
  {
    assert(transform.m == divisor.dim);
    assert(transform.m == extent.dim);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline DomainScaleTransform::DomainScaleTransform(
      const ScaleTransform<M, N, T>& rhs)
    : transform(rhs.transform), extent(rhs.extent), divisor(rhs.divisor)
  //----------------------------------------------------------------------------
  { }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__ inline DomainScaleTransform&
      DomainScaleTransform::operator=(const DomainScaleTransform& rhs)
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
  __LEGION_CUDA_HD__ inline DomainScaleTransform&
      DomainScaleTransform::operator=(const ScaleTransform<M, N, T>& rhs)
  //----------------------------------------------------------------------------
  {
    transform = rhs.transform;
    extent = rhs.extent;
    divisor = rhs.divisor;
    return *this;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline bool DomainScaleTransform::operator==(
      const DomainScaleTransform& rhs) const
  //----------------------------------------------------------------------------
  {
    if (transform != rhs.transform)
      return false;
    if (extent != rhs.extent)
      return false;
    if (divisor != rhs.divisor)
      return false;
    return true;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline bool DomainScaleTransform::operator!=(
      const DomainScaleTransform& rhs) const
  //----------------------------------------------------------------------------
  {
    return !(*this == rhs);
  }

  //----------------------------------------------------------------------------
  template<int M, int N, typename T>
  __LEGION_CUDA_HD__ inline DomainScaleTransform::operator ScaleTransform<
      M, N, T>(void) const
  //----------------------------------------------------------------------------
  {
    ScaleTransform<M, N, T> result;
    result.transform = transform;
    result.extent = extent;
    result.divisor = divisor;
    return result;
  }

  //----------------------------------------------------------------------------
  __LEGION_CUDA_HD__
  inline Domain DomainScaleTransform::operator[](const DomainPoint& p) const
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
  __LEGION_CUDA_HD__
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

}  // namespace Legion
