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

// Included from logical.h - do not include this directly

// Useful for IDEs
#include "legion/views/logical.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_materialized_did(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, MATERIALIZED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_reduction_did(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, REDUCTION_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_replicated_did(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, REPLICATED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_allreduce_did(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, ALLREDUCE_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_fill_did(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, FILL_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID LogicalView::encode_phi_did(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, PHI_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_materialized_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return (
          (LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC - 1)) ==
          MATERIALIZED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_reduction_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return (
          (LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC - 1)) ==
          REDUCTION_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_replicated_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return (
          (LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC - 1)) ==
          REPLICATED_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_allreduce_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return (
          (LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC - 1)) ==
          ALLREDUCE_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_individual_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return is_materialized_did(did) || is_reduction_did(did);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_collective_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return is_replicated_did(did) || is_allreduce_did(did);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_fill_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return (
          (LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC - 1)) ==
          FILL_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool LogicalView::is_phi_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return (
          (LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC - 1)) ==
          PHI_VIEW_DC);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_instance_view(void) const
    //--------------------------------------------------------------------------
    {
      return (
          is_materialized_did(did) || is_reduction_did(did) ||
          is_replicated_did(did) || is_allreduce_did(did));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_deferred_view(void) const
    //--------------------------------------------------------------------------
    {
      return (is_fill_did(did) || is_phi_did(did));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_materialized_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_reduction_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_replicated_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_replicated_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_allreduce_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_allreduce_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_individual_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_individual_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_collective_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_collective_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_fill_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_fill_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_phi_view(void) const
    //--------------------------------------------------------------------------
    {
      return is_phi_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::is_reduction_kind(void) const
    //--------------------------------------------------------------------------
    {
      return is_reduction_view() || is_allreduce_view();
    }

    //--------------------------------------------------------------------------
    inline void LogicalView::add_base_valid_ref(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      add_base_valid_ref_internal(source, cnt);
#else
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return;
      }
      add_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void LogicalView::add_nested_valid_ref(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      add_nested_valid_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return;
      }
      add_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::remove_base_valid_ref(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_base_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      return remove_base_valid_ref_internal(source, cnt);
#else
      int current = valid_references.load();
      legion_assert(current >= cnt);
      while (current > cnt)
      {
        int next = current - cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool LogicalView::remove_nested_valid_ref(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_nested_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      return remove_nested_valid_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = valid_references.load();
      legion_assert(current >= cnt);
      while (current > cnt)
      {
        int next = current - cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_valid_reference(cnt);
#endif
    }

  }  // namespace Internal
}  // namespace Legion
