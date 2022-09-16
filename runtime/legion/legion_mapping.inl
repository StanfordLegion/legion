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

// Included from legion_mapping.h - do not include this directly

// Useful for IDEs 
#include "legion/legion_mapping.h"

namespace Legion {
  namespace Mapping {

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    VariantID MapperRuntime::register_task_variant(MapperContext ctx,
                                          const TaskVariantRegistrar &registrar)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<T,TASK_PTR>);
      return register_task_variant(ctx, registrar, desc, NULL/*UDT*/,
                                   0/*sizeof(UDT)*/, true/*has return type*/);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    VariantID MapperRuntime::register_task_variant(MapperContext ctx,
                    const TaskVariantRegistrar &registrar, const UDT &user_data)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(
          LegionTaskWrapper::legion_udt_task_wrapper<T,UDT,TASK_PTR>);
      return register_task_variant(ctx, registrar, desc, &user_data, 
                                   sizeof(UDT), true/*has return type*/);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*)>
    VariantID MapperRuntime::register_task_variant(MapperContext ctx,
                                          const TaskVariantRegistrar &registrar)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
      return register_task_variant(ctx, registrar, desc, NULL/*UDT*/, 
                                   0/*sizeof(UDT)*/, false/*has return type*/);
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    VariantID MapperRuntime::register_task_variant(MapperContext ctx,
                    const TaskVariantRegistrar &registrar, const UDT &user_data)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(
          LegionTaskWrapper::legion_udt_task_wrapper<UDT,TASK_PTR>);
      return register_variant(ctx, registrar, desc, &user_data, sizeof(UDT),
                              false/*has return type*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceT<DIM,T> MapperRuntime::create_index_space(
                  MapperContext ctx, Rect<DIM,T> bounds, const char *prov) const
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> realm_is((Realm::IndexSpace<DIM,T>(bounds)));
      const Domain dom(realm_is);
      return IndexSpaceT<DIM,T>(create_index_space(ctx, dom,
       Legion::Internal::NT_TemplateHelper::template encode_tag<DIM,T>(),prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceT<DIM,T> MapperRuntime::create_index_space(
              MapperContext ctx, const std::vector<Point<DIM,T> > &points,
              const char *provenance) const
    //--------------------------------------------------------------------------
    {
      // C++ type system is dumb
      std::vector<Realm::Point<DIM,T> > realm_points(points.size());
      for (unsigned idx = 0; idx < points.size(); idx++)
        realm_points[idx] = points[idx];
      DomainT<DIM,T> realm_is((Realm::IndexSpace<DIM,T>(realm_points)));
      const Domain dom(realm_is);
      return IndexSpaceT<DIM,T>(create_index_space(ctx, dom,
        Internal::NT_TemplateHelper::template encode_tag<DIM,T>(), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceT<DIM,T> MapperRuntime::create_index_space(
                MapperContext ctx, const std::vector<Rect<DIM,T> > &rects,
                const char *provenance) const
    //--------------------------------------------------------------------------
    {
      // C++ type system is dumb
      std::vector<Realm::Rect<DIM,T> > realm_rects(rects.size());
      for (unsigned idx = 0; idx < rects.size(); idx++)
        realm_rects[idx] = rects[idx];
      DomainT<DIM,T> realm_is((Realm::IndexSpace<DIM,T>(realm_rects)));
      const Domain dom(realm_is);
      return IndexSpaceT<DIM,T>(create_index_space(ctx, dom,
        Internal::NT_TemplateHelper::template encode_tag<DIM,T>(), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceT<DIM,T> MapperRuntime::union_index_spaces(
          MapperContext ctx, const std::vector<IndexSpaceT<DIM,T> > &srcs,
          const char *provenance) const
    //--------------------------------------------------------------------------
    {
      // C++ type system is dumb
      std::vector<IndexSpace> sources(srcs.size());
      for (unsigned idx = 0; idx < sources.size(); idx++)
        sources[idx] = srcs[idx];
      return IndexSpaceT<DIM,T>(union_index_spaces(ctx, sources, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceT<DIM,T> MapperRuntime::intersect_index_spaces(
          MapperContext ctx, const std::vector<IndexSpaceT<DIM,T> > &srcs,
          const char *provenance) const
    //--------------------------------------------------------------------------
    {
      // C++ type system is dumb
      std::vector<IndexSpace> sources(srcs.size());
      for (unsigned idx = 0; idx < sources.size(); idx++)
        sources[idx] = srcs[idx];
      return IndexSpaceT<DIM,T>(intersect_index_spaces(ctx,sources,provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceT<DIM,T> MapperRuntime::subtract_index_spaces(
                         MapperContext ctx, IndexSpaceT<DIM,T> left,
                         IndexSpaceT<DIM,T> right, const char *provenance) const
    //--------------------------------------------------------------------------
    {
      return IndexSpaceT<DIM,T>(subtract_index_spaces(ctx, 
                              (IndexSpace)left, (IndexSpace)right), provenance);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline bool MapperRuntime::is_index_space_empty(MapperContext ctx,
                                                IndexSpaceT<DIM,T> handle) const
    //--------------------------------------------------------------------------
    {
      return is_index_space_empty(ctx, (IndexSpace)handle);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline bool MapperRuntime::index_spaces_overlap(MapperContext ctx,
                           IndexSpaceT<DIM,T> one, IndexSpaceT<DIM,T> two) const
    //--------------------------------------------------------------------------
    {
      return index_spaces_overlap(ctx, (IndexSpace)one, (IndexSpace)two);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline bool MapperRuntime::index_space_dominates(MapperContext ctx,
                        IndexSpaceT<DIM,T> left, IndexSpaceT<DIM,T> right) const
    //--------------------------------------------------------------------------
    {
      return index_space_dominates(ctx, (IndexSpace)left, (IndexSpace)right);
    }

    //--------------------------------------------------------------------------
    inline ProfilingRequest::ProfilingRequest(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    inline ProfilingRequest::~ProfilingRequest(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template <typename T>
    inline ProfilingRequest &ProfilingRequest::add_measurement(void)
    //--------------------------------------------------------------------------
    {
      requested_measurements.insert((ProfilingMeasurementID)T::ID);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool ProfilingRequest::empty(void) const
    //--------------------------------------------------------------------------
    {
      return requested_measurements.empty();
    }

    //--------------------------------------------------------------------------
    inline ProfilingResponse::ProfilingResponse(void)
    //--------------------------------------------------------------------------
      : realm_resp(NULL), overhead(NULL) 
    {
    }

    //--------------------------------------------------------------------------
    inline ProfilingResponse::~ProfilingResponse(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template <typename T>
    inline bool ProfilingResponse::has_measurement(void) const
    //--------------------------------------------------------------------------
    {
      // Realm measurements, all Legion measurements are specialized templates 
      if (realm_resp)
        return realm_resp->template has_measurement<T>();
      else
        return false;
    }

    //--------------------------------------------------------------------------
    template<>
    inline bool ProfilingResponse::has_measurement<
                             ProfilingMeasurements::RuntimeOverhead>(void) const
    //--------------------------------------------------------------------------
    {
      return (overhead != NULL);
    }

    //--------------------------------------------------------------------------
    template <typename T>
    inline T *ProfilingResponse::get_measurement(void) const
    //--------------------------------------------------------------------------
    {
      // Realm measurements, all Legion measurements are specialized templates
      if (realm_resp)
        return realm_resp->template get_measurement<T>();
      else
        return 0;
    }

    //--------------------------------------------------------------------------
    template <typename T>
    inline bool ProfilingResponse::get_measurement(T& result) const
    //--------------------------------------------------------------------------
    {
      if (realm_resp)
        return realm_resp->template get_measurement<T>(result);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    template<>
    inline ProfilingMeasurements::RuntimeOverhead* 
            ProfilingResponse::get_measurement<
                             ProfilingMeasurements::RuntimeOverhead>(void) const 
    //--------------------------------------------------------------------------
    {
      // Always produce a copy for the mapper to own
      return new ProfilingMeasurements::RuntimeOverhead(*overhead);
    }

    //--------------------------------------------------------------------------
    template<>
    inline bool ProfilingResponse::get_measurement<
                  ProfilingMeasurements::RuntimeOverhead>(
                           ProfilingMeasurements::RuntimeOverhead& result) const
    //--------------------------------------------------------------------------
    {
      result = *overhead;
      return true;
    }

    namespace ProfilingMeasurements {
      //------------------------------------------------------------------------
      inline RuntimeOverhead::RuntimeOverhead(void)
        : application_time(0), runtime_time(0), wait_time(0)
      //------------------------------------------------------------------------
      {
      }
    };
  };
};

