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

#ifndef __LEGION_TRANSFORMS_H__
#define __LEGION_TRANSFORMS_H__

#include "legion/api/geometry.h"

namespace Legion {

  /**
   * \class AffineTransform
   * An affine transform is used to transform points in one
   * coordinate space into points in another coordinate space
   * using the basic Ax + b transformation, where A is a
   * transform matrix and b is an offset vector
   */
  template<int M, int N, typename T = coord_t>
  struct AffineTransform {
  private:
    static_assert(M > 0, "M must be positive");
    static_assert(N > 0, "N must be positive");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    AffineTransform(void);  // default to identity transform
    // allow type coercions where possible
    template<typename T2>
    __LEGION_CUDA_HD__ AffineTransform(const AffineTransform<M, N, T2>& rhs);
    template<typename T2, typename T3>
    __LEGION_CUDA_HD__ AffineTransform(
        const Transform<M, N, T2> transform, const Point<M, T3> offset);
  public:
    template<typename T2>
    __LEGION_CUDA_HD__ AffineTransform<M, N, T>& operator=(
        const AffineTransform<M, N, T2>& rhs);
  public:
    // Apply the transformation to a point
    template<typename T2>
    __LEGION_CUDA_HD__ Point<M, T> operator[](const Point<N, T2> point) const;
    // Compose the transform with another transform
    template<int P>
    __LEGION_CUDA_HD__ AffineTransform<M, P, T> operator()(
        const AffineTransform<N, P, T>& rhs) const;
    // Test whether this is the identity transform
    __LEGION_CUDA_HD__
    bool is_identity(void) const;
  public:
    // Transform = Ax + b
    Transform<M, N, T> transform;  // A
    Point<M, T> offset;            // b
  };

  /**
   * \class ScaleTransform
   * A scale transform is a used to do a projection transform
   * that converts a point in one coordinate space into a range
   * in another coordinate system using the transform:
   *    [y0, y1] = Ax + [b, c]
   *              ------------
   *                   d
   *  where all lower case letters are points and A is
   *  transform matrix. Note that by making b == c then
   *  we can make this a one-to-one point mapping.
   */
  template<int M, int N, typename T = coord_t>
  struct ScaleTransform {
  private:
    static_assert(M > 0, "M must be positive");
    static_assert(M > 0, "N must be positive");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    ScaleTransform(void);  // default to identity transform
    // allow type coercions where possible
    template<typename T2>
    __LEGION_CUDA_HD__ ScaleTransform(const ScaleTransform<M, N, T2>& rhs);
    template<typename T2, typename T3, typename T4>
    __LEGION_CUDA_HD__ ScaleTransform(
        const Transform<M, N, T2> transform, const Rect<M, T3> extent,
        const Point<M, T4> divisor);
  public:
    template<typename T2>
    __LEGION_CUDA_HD__ ScaleTransform<M, N, T>& operator=(
        const ScaleTransform<M, N, T2>& rhs);
  public:
    // Apply the transformation to a point
    template<typename T2>
    __LEGION_CUDA_HD__ Rect<M, T> operator[](const Point<N, T2> point) const;
    // Test whether this is the identity transform
    __LEGION_CUDA_HD__
    bool is_identity(void) const;
  public:
    Transform<M, N, T> transform;  // A
    Rect<M, T> extent;             // [b=lo, c=hi]
    Point<M, T> divisor;           // d
  };

  // If we've got c++11 we can just include this directly

  /**
   * \class DomainTransform
   * A type-erased version of a Transform for removing template
   * parameters from a Transform object
   */
  class DomainTransform {
  public:
    __LEGION_CUDA_HD__
    DomainTransform(void);
    __LEGION_CUDA_HD__
    DomainTransform(const DomainTransform& rhs);
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ DomainTransform(const Transform<M, N, T>& rhs);
  public:
    __LEGION_CUDA_HD__
    DomainTransform& operator=(const DomainTransform& rhs);
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ DomainTransform& operator=(
        const Transform<M, N, T>& rhs);
    __LEGION_CUDA_HD__
    bool operator==(const DomainTransform& rhs) const;
    __LEGION_CUDA_HD__
    bool operator!=(const DomainTransform& rhs) const;
  public:
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ operator Transform<M, N, T>(void) const;
  public:
    __LEGION_CUDA_HD__
    DomainPoint operator*(const DomainPoint& p) const;
    __LEGION_CUDA_HD__
    Domain operator*(const Domain& domain) const;
    __LEGION_CUDA_HD__
    DomainTransform operator*(const DomainTransform& transform) const;
  public:
    __LEGION_CUDA_HD__
    bool is_identity(void) const;
  public:
    int m, n;
    coord_t matrix[LEGION_MAX_DIM * LEGION_MAX_DIM];
  };

  /**
   * \class DomainAffineTransform
   * A type-erased version of an AffineTransform for removing
   * template parameters from an AffineTransform type
   */
  class DomainAffineTransform {
  public:
    __LEGION_CUDA_HD__
    DomainAffineTransform(void);
    __LEGION_CUDA_HD__
    DomainAffineTransform(const DomainAffineTransform& rhs);
    __LEGION_CUDA_HD__
    DomainAffineTransform(const DomainTransform& t, const DomainPoint& p);
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ DomainAffineTransform(
        const AffineTransform<M, N, T>& transform);
  public:
    __LEGION_CUDA_HD__
    DomainAffineTransform& operator=(const DomainAffineTransform& rhs);
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ DomainAffineTransform& operator=(
        const AffineTransform<M, N, T>& rhs);
    __LEGION_CUDA_HD__
    bool operator==(const DomainAffineTransform& rhs) const;
    __LEGION_CUDA_HD__
    bool operator!=(const DomainAffineTransform& rhs) const;
  public:
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ operator AffineTransform<M, N, T>(void) const;
  public:
    // Apply the transformation to a point
    __LEGION_CUDA_HD__
    DomainPoint operator[](const DomainPoint& p) const;
    // Test for the identity
    __LEGION_CUDA_HD__
    bool is_identity(void) const;
  public:
    DomainTransform transform;
    DomainPoint offset;
  };

  /**
   * \class DomainScaleTransform
   * A type-erased version of a ScaleTransform for removing
   * template parameters from a ScaleTransform type
   */
  class DomainScaleTransform {
  public:
    __LEGION_CUDA_HD__
    DomainScaleTransform(void);
    __LEGION_CUDA_HD__
    DomainScaleTransform(const DomainScaleTransform& rhs);
    __LEGION_CUDA_HD__
    DomainScaleTransform(
        const DomainTransform& transform, const Domain& extent,
        const DomainPoint& divisor);
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ DomainScaleTransform(
        const ScaleTransform<M, N, T>& transform);
  public:
    __LEGION_CUDA_HD__
    DomainScaleTransform& operator=(const DomainScaleTransform& rhs);
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ DomainScaleTransform& operator=(
        const ScaleTransform<M, N, T>& rhs);
    __LEGION_CUDA_HD__
    bool operator==(const DomainScaleTransform& rhs) const;
    __LEGION_CUDA_HD__
    bool operator!=(const DomainScaleTransform& rhs) const;
  public:
    template<int M, int N, typename T>
    __LEGION_CUDA_HD__ operator ScaleTransform<M, N, T>(void) const;
  public:
    // Apply the transformation to a point
    __LEGION_CUDA_HD__
    Domain operator[](const DomainPoint& p) const;
    // Test for the identity
    __LEGION_CUDA_HD__
    bool is_identity(void) const;
  public:
    DomainTransform transform;
    Domain extent;
    DomainPoint divisor;
  };

}  // namespace Legion

#include "legion/api/transforms.inl"

#endif  // __LEGION_TRANSFORMS_H__
