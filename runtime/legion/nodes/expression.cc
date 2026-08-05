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

#include "legion/nodes/expression.h"
#include "legion/nodes/index.h"
#include "legion/nodes/kdtree.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Index Space Expression
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(LocalLock& lock)
      : type_tag(0), expr_id(0), expr_lock(lock), sparsity_map_kd_tree(nullptr),
        volume(0), has_volume(false), empty(false), has_empty(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(TypeTag tag, LocalLock& lock)
      : type_tag(tag), expr_id(runtime->get_unique_index_space_expr_id()),
        expr_lock(lock), sparsity_map_kd_tree(nullptr), volume(0),
        has_volume(false), empty(false), has_empty(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexSpaceExpression::IndexSpaceExpression(
        TypeTag tag, IndexSpaceExprID id, LocalLock& lock)
      : type_tag(tag), expr_id(id), expr_lock(lock),
        sparsity_map_kd_tree(nullptr), volume(0), has_volume(false),
        empty(false), has_empty(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexSpaceExpression::~IndexSpaceExpression(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(derived_operations.empty());
      legion_assert(retired_canonical.load() == nullptr);
      if (sparsity_map_kd_tree != nullptr)
        delete sparsity_map_kd_tree;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceExpression::TightenIndexSpaceArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->tighten_index_space();
      if (proxy_dc->remove_base_resource_ref(META_TASK_REF))
        delete proxy_this;
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID IndexSpaceExpression::get_owner_space(
        IndexSpaceExprID expr_id)
    //--------------------------------------------------------------------------
    {
      return (expr_id % runtime->runtime_stride);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceExpression::add_derived_operation(IndexSpaceOperation* op)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(expr_lock);
      legion_assert(derived_operations.find(op) == derived_operations.end());
      derived_operations.insert(op);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceExpression::remove_derived_operation(IndexSpaceOperation* op)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(expr_lock);
      legion_assert(
          derived_operations.empty() ||
          (derived_operations.find(op) != derived_operations.end()));
      derived_operations.erase(op);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceExpression::invalidate_derived_operations(void)
    //--------------------------------------------------------------------------
    {
      // Traverse upwards for any derived operations and invalidate them
      const size_t previous_operations =
          implicit_reference_tracker->count_invalid_operations();
      {
        AutoLock e_lock(expr_lock);
        for (IndexSpaceOperation* const operation : derived_operations)
          if (operation->own_invalidation())
            // Note the REGION_TREE_REF keeps this alive until the implicit
            // reference counter is able to reclaim it
            ImplicitReferenceTracker::record_invalid_operation(operation);
        derived_operations.clear();
      }
      for (unsigned idx = previous_operations;
           idx < implicit_reference_tracker->count_invalid_operations(); idx++)
      {
        IndexSpaceOperation* operation =
            implicit_reference_tracker->get_invalid_operation(idx);
        operation->invalidate_operation(this);
      }
      implicit_reference_tracker->invalidate_operations();
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceExpression::test_intersection_nonblocking(
        IndexSpaceExpression* other, ApEvent& precondition, bool second)
    //--------------------------------------------------------------------------
    {
      if (second)
      {
        // We've got two non pending expressions, so we can just test them
        IndexSpaceExpression* overlap =
            runtime->intersect_index_spaces(this, other);
        return !overlap->is_empty();
      }
      else
      {
        // First time through, we're not pending so keep going
        return other->test_intersection_nonblocking(
            this, precondition, true /*second*/);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* IndexSpaceExpression::get_canonical_expression(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_valid());
      // Small helper struct to help with making sure that it is safe
      // to access the canonical atomic
      struct AutoCanonical {
      public:
        AutoCanonical(IndexSpaceExpression* own) : owner(own)
        {
          // Must be done in this order for safety
          owner->canonical_semaphore++;
          expr = owner->canonical.load();
        }
        ~AutoCanonical(void)
        {
          const unsigned remaining = --owner->canonical_semaphore;
          legion_assert(remaining != UINT_MAX);  // check for underflow
          if (remaining > 0)
            return;
          // We were the last one out so reclaim any retired canonical
          // expression. Note that we cannot key off our own 'expr' here
          // since we might not have been the one that replaced it. We also
          // have to test the semaphore again after reading it: a thread
          // that entered after our decrement can have loaded a canonical
          // expression that a third thread has since retired, and that one
          // is not safe to reclaim out from underneath them. Any thread we
          // see here is going to make this same test on its way out, so
          // whoever ends up being last will do the reclamation.
          IndexSpaceExpression* stale = owner->retired_canonical.load();
          if ((stale != nullptr) && (owner->canonical_semaphore.load() == 0) &&
              owner->retired_canonical.compare_exchange_strong(
                  stale, nullptr) &&
              stale->remove_canonical_reference(owner->get_distributed_id()))
            delete stale;
        }
      public:
        IndexSpaceExpression* const owner;
        IndexSpaceExpression* expr = nullptr;
      };
      // Check for the common case where we are already our own canonical
      // expression first. We never dereference the result on that path and
      // our caller is holding a live reference on us, so there is nothing
      // here that the semaphore needs to protect and we can skip it. Note
      // that we still have to go the slow way if we have a retired
      // canonical expression outstanding, since this is a sticky state and
      // nobody would ever come back to reclaim it otherwise
      IndexSpaceExpression* expr = canonical.load();
      if ((expr == this) && (retired_canonical.load() == nullptr))
        return expr;
      AutoCanonical canon(this);
      // Discard the peek above and take the value the semaphore protects
      expr = canon.expr;
      if (expr != nullptr)
      {
        // If we're our own canonical expression then assume we're
        // still alive and don't need to add a live expression
        if (expr == this)
          return expr;
        // If we're not our own canonical expression, we need to make sure
        // that it is still alive and can be used
        if (expr->try_add_live_reference())
          return expr;
        // Fall through and compute a new canonical expression
      }
      expr = runtime->find_canonical_expression(this);
      if (expr == this)
      {
        // If we're our own canonical expression then the runtime didn't
        // give us a reference to ourself
        // Hang on to our reference on the previous canonical expression
        // until the last thread out of here can prove that nobody is still
        // holding a pointer that they loaded out of 'canonical'
        IndexSpaceExpression* prev = canonical.exchange(expr);
        if ((prev != nullptr) && (prev != expr))
          legion_no_skip_assert(retired_canonical.exchange(prev) == nullptr);
        return expr;
      }
      // If the canonical expression is not ourself, then the region tree
      // runtime has given us a live reference back on it so we know it
      // can't be collected, but we need to update the canonical result
      // and add a nested reference if we're the first ones to perform
      // the update. Note that we add the nested resource reference before
      // we publish the pointer so that any other thread which loads it out
      // of 'canonical' sees an expression that we are already keeping alive
      const DistributedID did = get_distributed_id();
      expr->add_canonical_reference(did);
      IndexSpaceExpression* prev = canonical.exchange(expr);
      if (prev != expr)
      {
        // Hang on to our reference on the previous canonical expression
        // until the last thread out of here can prove that nobody is still
        // holding a pointer that they loaded out of 'canonical'
        if (prev != nullptr)
          legion_no_skip_assert(retired_canonical.exchange(prev) == nullptr);
      }
      else
        // Someone else stored this same result before we did and they are
        // holding the canonical reference for it, so we can remove the
        // extra one that we just added. Note that this can never be the
        // last reference precisely because they are still holding theirs:
        // reclaiming it would need the semaphore to be at zero and we are
        // standing in it
        legion_no_skip_assert(!expr->remove_canonical_reference(did));
      return expr;
    }

    //--------------------------------------------------------------------------
    /*static*/ IndexSpaceExpression* IndexSpaceExpression::unpack_expression(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Handle the special case where this is a local index space expression
      bool is_local;
      derez.deserialize(is_local);
      if (is_local)
      {
        IndexSpaceExpression* result;
        derez.deserialize(result);
        if (source != runtime->address_space)
        {
          IndexSpaceOperation* op =
              legion_safe_cast<IndexSpaceOperation*>(result);
          if (!op->try_add_live_reference())
            std::abort();
          op->unpack_global_ref(derez);
        }
        else
          // LIVE_EXPR_REF was added by the pack_expression call
          ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      bool is_index_space;
      derez.deserialize(is_index_space);
      // If this is an index space it is easy
      if (is_index_space)
      {
        IndexSpace handle;
        derez.deserialize(handle);
        IndexSpaceNode* node = runtime->get_node(handle);
        if (!node->try_add_live_reference())
          std::abort();
        // Now we can unpack the global expression reference
        node->unpack_global_ref(derez);
        return node;
      }
      else
      {
        IndexSpaceExprID remote_expr_id;
        derez.deserialize(remote_expr_id);
        IndexSpaceExpression* result =
            runtime->find_or_create_remote_expression(
                remote_expr_id, derez, source);
        IndexSpaceOperation* op =
            legion_safe_cast<IndexSpaceOperation*>(result);
        if (!result->try_add_live_reference())
          std::abort();
        // Unpack the global reference that we had
        op->unpack_global_ref(derez);
        return result;
      }
    }

    /////////////////////////////////////////////////////////////
    // Index Space Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceOperation::IndexSpaceOperation(TypeTag tag, OperationKind kind)
      : IndexSpaceExpression(tag, inter_lock),
        DistributedCollectable(LEGION_DISTRIBUTED_HELP_ENCODE(
            runtime->get_available_distributed_id(), INDEX_EXPR_NODE_DC)),
        origin_expr(this), op_kind(kind)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info(
          "GC Index Expr %lld %d %lld", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space, expr_id);
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpaceOperation::IndexSpaceOperation(
        TypeTag tag, IndexSpaceExprID eid, DistributedID did,
        IndexSpaceOperation* origin)
      : IndexSpaceExpression(tag, eid, inter_lock), DistributedCollectable(did),
        origin_expr(origin), op_kind(REMOTE_EXPRESSION_KIND)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner());
#ifdef LEGION_GC
      log_garbage.info(
          "GC Index Expr %lld %d %lld", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space, expr_id);
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpaceOperation::~IndexSpaceOperation(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        runtime->unregister_remote_expression(expr_id);
      // Invalidate any derived operations
      invalidate_derived_operations();
      // Remove this operation from the region tree
      remove_operation();
      IndexSpaceExpression* canon = canonical.load();
      if (canon != nullptr)
      {
        if (canon == this)
          runtime->remove_canonical_expression(this);
        else if (canon->remove_canonical_reference(did))
          delete canon;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::try_add_live_reference(void)
    //--------------------------------------------------------------------------
    {
      if (check_global_and_increment(LIVE_EXPR_REF))
      {
        ImplicitReferenceTracker::record_live_expression(this);
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_base_expression_reference(
        ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      legion_assert(has_gc_reference());
      add_base_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_nested_expression_reference(
        DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      legion_assert(has_gc_reference());
      add_nested_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_base_expression_reference(
        ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_base_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_nested_expression_reference(
        DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceOperation::add_tree_expression_reference(
        DistributedID id, unsigned count)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(id, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::remove_tree_expression_reference(
        DistributedID id, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(id, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceOperation::own_invalidation(void)
    //--------------------------------------------------------------------------
    {
      return !invalidated.exchange(true);
    }

    /////////////////////////////////////////////////////////////
    // Expression Trie Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExpressionTrieNode::ExpressionTrieNode(
        unsigned d, IndexSpaceExprID id, IndexSpaceExpression* op)
      : depth(d), expr(id), local_operation(op)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ExpressionTrieNode::~ExpressionTrieNode(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool ExpressionTrieNode::find_operation(
        const std::vector<IndexSpaceExpression*>& expressions,
        IndexSpaceExpression*& result, ExpressionTrieNode*& last)
    //--------------------------------------------------------------------------
    {
      legion_assert(depth < expressions.size());
      legion_assert(expressions[depth]->expr_id == expr);  // these should match
      // Three cases here
      if (expressions.size() == (depth + 1))
      {
        // We're the node that should have the operation
        // Check to see if we've made the operation yet
        if (local_operation != nullptr)
        {
          result = local_operation;
          return true;
        }
        last = this;
        return false;
      }
      else if (expressions.size() == (depth + 2))
      {
        // The next node should have the operation, but we might be
        // storing it until it actually gets made
        // See if we already have it or we have the next trie node
        ExpressionTrieNode* next = nullptr;
        const IndexSpaceExprID target_expr = expressions.back()->expr_id;
        {
          AutoLock t_lock(trie_lock, false /*exclusive*/);
          std::map<IndexSpaceExprID, IndexSpaceExpression*>::const_iterator
              op_finder = operations.find(target_expr);
          if (op_finder != operations.end())
          {
            result = op_finder->second;
            return true;
          }
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              node_finder = nodes.find(target_expr);
          if (node_finder != nodes.end())
            next = node_finder->second;
        }
        // Didn't find either, retake the lock in exclusive mode and then
        // see if we lost the race, if not make the operation or
        if (next == nullptr)
        {
          AutoLock t_lock(trie_lock);
          std::map<IndexSpaceExprID, IndexSpaceExpression*>::const_iterator
              op_finder = operations.find(target_expr);
          if (op_finder != operations.end())
          {
            result = op_finder->second;
            return true;
          }
          // Still don't have the op
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              node_finder = nodes.find(target_expr);
          if (node_finder == nodes.end())
          {
            last = this;
            return false;
          }
          else
            next = node_finder->second;
        }
        legion_assert(next != nullptr);
        return next->find_operation(expressions, result, last);
      }
      else
      {
        // Intermediate case
        // See if we have the next node, or if we have to make it
        ExpressionTrieNode* next = nullptr;
        const IndexSpaceExprID target_expr = expressions[depth + 1]->expr_id;
        {
          AutoLock t_lock(trie_lock, false /*exclusive*/);
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              finder = nodes.find(target_expr);
          if (finder != nodes.end())
            next = finder->second;
        }
        // Still don't have it so we have to try and make it
        if (next == nullptr)
        {
          AutoLock t_lock(trie_lock);
          // See if we lost the race
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              finder = nodes.find(target_expr);
          if (finder == nodes.end())
          {
            // We have to make the next node, also check to see if we
            // already made an operation expression for it or not
            std::map<IndexSpaceExprID, IndexSpaceExpression*>::iterator
                op_finder = operations.find(target_expr);
            if (op_finder != operations.end())
            {
              next = new ExpressionTrieNode(
                  depth + 1, target_expr, op_finder->second);
              operations.erase(op_finder);
            }
            else
              next = new ExpressionTrieNode(depth + 1, target_expr);
            nodes[target_expr] = next;
          }
          else  // lost the race
            next = finder->second;
        }
        legion_assert(next != nullptr);
        return next->find_operation(expressions, result, last);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* ExpressionTrieNode::find_or_create_operation(
        const std::vector<IndexSpaceExpression*>& expressions,
        OperationCreator& creator)
    //--------------------------------------------------------------------------
    {
      legion_assert(depth < expressions.size());
      legion_assert(expressions[depth]->expr_id == expr);  // these should match
      // Three cases here
      if (expressions.size() == (depth + 1))
      {
        // We're the node that should have the operation
        // Check to see if we've made the operation yet
        if ((local_operation != nullptr) &&
            local_operation->try_add_live_reference())
          return local_operation;
        // Operation doesn't exist yet, retake the lock and try to make it
        AutoLock t_lock(trie_lock);
        if ((local_operation != nullptr) &&
            local_operation->try_add_live_reference())
          return local_operation;
        local_operation = creator.consume();
        if (!local_operation->try_add_live_reference())
          std::abort();  // should never hit this
        return local_operation;
      }
      else if (expressions.size() == (depth + 2))
      {
        // The next node should have the operation, but we might be
        // storing it until it actually gets made
        // See if we already have it or we have the next trie node
        ExpressionTrieNode* next = nullptr;
        const IndexSpaceExprID target_expr = expressions.back()->expr_id;
        {
          AutoLock t_lock(trie_lock, false /*exclusive*/);
          std::map<IndexSpaceExprID, IndexSpaceExpression*>::const_iterator
              op_finder = operations.find(target_expr);
          if ((op_finder != operations.end()) &&
              op_finder->second->try_add_live_reference())
            return op_finder->second;
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              node_finder = nodes.find(target_expr);
          if (node_finder != nodes.end())
            next = node_finder->second;
        }
        // Didn't find either, retake the lock in exclusive mode and then
        // see if we lost the race, if not make the operation or
        if (next == nullptr)
        {
          AutoLock t_lock(trie_lock);
          std::map<IndexSpaceExprID, IndexSpaceExpression*>::const_iterator
              op_finder = operations.find(target_expr);
          if ((op_finder != operations.end()) &&
              op_finder->second->try_add_live_reference())
            return op_finder->second;
          // Still don't have the op
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              node_finder = nodes.find(target_expr);
          if (node_finder == nodes.end())
          {
            // Didn't find the sub-node, so make the operation here
            IndexSpaceExpression* result = creator.consume();
            operations[target_expr] = result;
            if (!result->try_add_live_reference())
              std::abort();  // should never hit this
            return result;
          }
          else
            next = node_finder->second;
        }
        legion_assert(next != nullptr);
        return next->find_or_create_operation(expressions, creator);
      }
      else
      {
        // Intermediate case
        // See if we have the next node, or if we have to make it
        ExpressionTrieNode* next = nullptr;
        const IndexSpaceExprID target_expr = expressions[depth + 1]->expr_id;
        {
          AutoLock t_lock(trie_lock, false /*exclusive*/);
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              finder = nodes.find(target_expr);
          if (finder != nodes.end())
            next = finder->second;
        }
        // Still don't have it so we have to try and make it
        if (next == nullptr)
        {
          AutoLock t_lock(trie_lock);
          // See if we lost the race
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              finder = nodes.find(target_expr);
          if (finder == nodes.end())
          {
            // We have to make the next node, also check to see if we
            // already made an operation expression for it or not
            std::map<IndexSpaceExprID, IndexSpaceExpression*>::iterator
                op_finder = operations.find(target_expr);
            if (op_finder != operations.end())
            {
              next = new ExpressionTrieNode(
                  depth + 1, target_expr, op_finder->second);
              operations.erase(op_finder);
            }
            else
              next = new ExpressionTrieNode(depth + 1, target_expr);
            nodes[target_expr] = next;
          }
          else  // lost the race
            next = finder->second;
        }
        legion_assert(next != nullptr);
        return next->find_or_create_operation(expressions, creator);
      }
    }

    //--------------------------------------------------------------------------
    bool ExpressionTrieNode::remove_operation(
        const std::vector<IndexSpaceExpression*>& expressions)
    //--------------------------------------------------------------------------
    {
      legion_assert(depth < expressions.size());
      legion_assert(expressions[depth]->expr_id == expr);  // these should match
      // No need for locks here, we're protected by the big lock at the top
      // Three cases here
      if (expressions.size() == (depth + 1))
      {
        // Simple case, clear our local operation
        local_operation = nullptr;
      }
      else if (expressions.size() == (depth + 2))
      {
        // See if we should continue traversing or if we have the operation
        const IndexSpaceExprID target_expr = expressions.back()->expr_id;
        std::map<IndexSpaceExprID, IndexSpaceExpression*>::iterator op_finder =
            operations.find(target_expr);
        if (op_finder == operations.end())
        {
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::iterator
              node_finder = nodes.find(target_expr);
          legion_assert(node_finder != nodes.end());
          if (node_finder->second->remove_operation(expressions))
          {
            delete node_finder->second;
            nodes.erase(node_finder);
          }
        }
        else
          operations.erase(op_finder);
      }
      else
      {
        const IndexSpaceExprID target_expr = expressions[depth + 1]->expr_id;
        std::map<IndexSpaceExprID, ExpressionTrieNode*>::iterator finder =
            nodes.find(target_expr);
        legion_assert(finder != nodes.end());
        if (finder->second->remove_operation(expressions))
        {
          delete finder->second;
          nodes.erase(finder);
        }
      }
      if (local_operation != nullptr)
        return false;
      if (!operations.empty())
        return false;
      if (!nodes.empty())
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    /*static*/ IndexSpaceOperation*
        InternalExpressionCreator::create_with_domain(
            TypeTag tag, const Domain& dom)
    //--------------------------------------------------------------------------
    {
      InternalExpressionCreator creator(tag, dom);
      creator.create_operation();

      IndexSpaceOperation* out = creator.result;
      if (!out->try_add_live_reference())
        std::abort();
      return out;
    }

  }  // namespace Internal
}  // namespace Legion
