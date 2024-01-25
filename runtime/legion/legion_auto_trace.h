/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_AUTO_TRACE_H__
#define __LEGION_AUTO_TRACE_H__

#include "legion.h"
#include "legion/legion_context.h"
#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/suffix_tree.h"
#include "legion/trie.h"

#include <queue>

template<>
struct std::hash<Legion::Internal::Murmur3Hasher::Hash> {
  std::size_t operator()(const Legion::Internal::Murmur3Hasher::Hash& h) const noexcept {
    return h.x ^ (h.y << 1);
  }
};

template<>
struct std::equal_to<Legion::Internal::Murmur3Hasher::Hash> {
  constexpr bool operator()(const Legion::Internal::Murmur3Hasher::Hash& lhs,
                            const Legion::Internal::Murmur3Hasher::Hash& rhs) const
  {
      return lhs.x == rhs.x && lhs.y == rhs.y;
  }
};

namespace Legion {
  namespace Internal {

    // Declare all of the necessary loggers.
    LEGION_EXTERN_LOGGER_DECLARATIONS

    // Forward declarations.
    class BatchedTraceIdentifier;
    class TraceOccurrenceWatcher;
    class TraceReplayer;

    // TraceHashHelper is a utility class to hash Operations.
    class TraceHashHelper {
    public:
      TraceHashHelper();
      Murmur3Hasher::Hash hash(Operation* op);
      void hash(TaskOp* op);
      void hash(FillOp* op);
      void hash(FenceOp* op);
      void hash(CopyOp* op);
      void hash(AllReduceOp* op);
      // TODO (rohany): DiscardOp.
      void hash(const RegionRequirement& req);
      void hash(const LogicalRegion& region);
    private:
      Murmur3Hasher hasher;
    };

    // TraceProcessingJobExecutor is an interface that hides the details
    // of executing and waiting for async trace processing jobs. It is
    // implemented by the InnerContext and ReplicateContext to handle
    // standard and control-replicated executions separately.
    class TraceProcessingJobExecutor {
    public:
      virtual RtEvent enqueue_task(
        const InnerContext::AutoTraceProcessRepeatsArgs& args,
        size_t opidx,
        bool wait
      ) = 0;
      virtual RtEvent poll_pending_tasks(
        size_t opidx,
        bool must_pop
      ) = 0;
    };

    // BatchedTraceIdentifier batches up operations until a given
    // size is hit, and then computes repeated substrings within
    // the batch of operations.
    class BatchedTraceIdentifier {
    public:
      BatchedTraceIdentifier(
        TraceProcessingJobExecutor* executor,
        TraceOccurrenceWatcher& watcher,
        size_t batchsize,  // Number of operations batched at once.
        size_t max_add, // Maximum number of traces to add to the watcher at once.
        size_t max_inflight_requests, // Maximum number of async jobs in flight
        bool wait_on_async_job, // Whether to wait on concurrent meta tasks
        size_t min_trace_length // Minimum trace length to identify.
      );
      void process(Murmur3Hasher::Hash hash, size_t opidx);
    private:
      // We need a runtime here in order to launch meta tasks.
      TraceProcessingJobExecutor* executor;
      std::vector<Murmur3Hasher::Hash> hashes;
      TraceOccurrenceWatcher& watcher;
      size_t batchsize;
      size_t max_add;
      size_t min_trace_length;

      // InFlightProcessingRequest represents a currently executing
      // offline string processing request. When the BatchedTraceIdentifier
      // launches a new meta task, it will register it inside the
      // in_flight_requests queue.
      struct InFlightProcessingRequest {
        std::vector<Murmur3Hasher::Hash> hashes;
        RtEvent finish_event;
        // Where the meta task should place the result of
        // the offline computation.
        std::vector<NonOverlappingRepeatsResult> result;
        bool completed = false;
      };
      std::list<InFlightProcessingRequest> jobs_in_flight;
      size_t max_in_flight_requests;
      bool wait_on_async_job;
    };

    // TraceOccurrenceWatcher tracks how many times inserted traces
    // have occured in the operation stream.
    class TraceOccurrenceWatcher {
    public:
      TraceOccurrenceWatcher(TraceReplayer& replayer, size_t visit_threshold);
      void process(Murmur3Hasher::Hash hash, size_t opidx);

      template<typename T>
      void insert(T start, T end, size_t opidx);
      template<typename T>
      bool prefix(T start, T end);

      // Clear invalidates all active pointers in the watcher.
      void clear() { this->active_pointers.clear(); }
    private:
      // Reference to a TraceReplayer to dump traces into.
      TraceReplayer& replayer;

      struct TraceMeta {
        // Needs to be default constructable.
        TraceMeta() : opidx(0) { }
        TraceMeta(size_t opidx_) : opidx(opidx_) { }
        // The opidx that this trace was inserted at.
        size_t opidx;
        // The occurrence watcher will only maintain the number
        // of visits. I don't think that we need to do decaying visits
        // here, though we might want to lower the amount of traces that
        // get committed to the replayer.
        size_t visits = 0;
        // completed marks whether this trace has moved
        // from the "watched" state to the "committed" state.
        // Once a trace has been completed, it will not be
        // returned from complete() anymore.
        bool completed = false;
        // The opidx that this trace was previously visited at.
        size_t previous_visited_opidx = 0;
      };
      Trie<Murmur3Hasher::Hash, TraceMeta> trie;
      size_t visit_threshold;

      // TriePointer maintains an active trace being
      // traversed in the watcher's trie.
      class TriePointer {
      public:
        TriePointer(TrieNode<Murmur3Hasher::Hash, TraceMeta>* node_, size_t opidx_)
          : node(node_), opidx(opidx_), depth(0) { }
        bool advance(Murmur3Hasher::Hash token);
        bool complete();
      public:
        TrieNode<Murmur3Hasher::Hash, TraceMeta>* node;
        size_t opidx;
        size_t depth;
      };
      // All currently active pointers that need advancing.
      std::vector<TriePointer> active_pointers;
    };

    // OperationExecutor is a virtual class to be used by the
    // TraceReplayer to shim out internal logic for actually
    // starting and ending traces and issuing operations.
    class OperationExecutor {
    public:
      virtual TraceID get_fresh_trace_id() = 0;
      virtual void issue_begin_trace(TraceID id) = 0;
      virtual void issue_end_trace(TraceID id) = 0;
      virtual bool issue_operation(Operation* op, bool unordered = false, bool outermost = true) = 0;
    };

    // TraceReplayer handles actually buffering and replaying committed traces.
    class TraceReplayer {
    public:
      TraceReplayer(OperationExecutor* executor_)
        : executor(executor_), operation_start_idx(0) { }

      // Enqueue a new operation, which has the given hash.
      void process(Operation* op, Murmur3Hasher::Hash hash, size_t opidx);
      void process_trace_noop(Operation* op);
      // Flush all pending operations out of the TraceReplayer. Accepts
      // the current opidx, used for scoring potentially replayed traces.
      void flush(size_t opidx);

      // Insert a new trace into the TraceReplayer.
      template<typename T>
      void insert(T start, T end, size_t opidx);
      // See if the chosen string is a prefix of a string contained
      // in the TraceReplayer.
      template<typename T>
      bool prefix(T start, T end);
    private:
      // Indirection layer to issue operations down to the underlying
      // runtime system.
      OperationExecutor* executor;

      struct TraceMeta {
        // TraceMeta's need to be default constructable.
        TraceMeta() {}
        TraceMeta(size_t opidx_, size_t length_)
          : opidx(opidx_), length(length_), last_visited_opidx(0),
            decaying_visits(0), replays(0),
            last_idempotent_visit_opidx(0),
            decaying_idempotent_visits(0.0), tid(0) { }
        // opidx that this trace was inserted at.
        size_t opidx;
        // length of the trace. This is used for scoring only.
        size_t length;
        // Fields for maintaining a decaying visit count.
        size_t last_visited_opidx;
        double decaying_visits;
        // Number of times the trace has been replayed.
        size_t replays;
        // Number of times the trace has been visited in
        // an idempotent manner (tracked in a decaying manner).
        size_t last_idempotent_visit_opidx;
        double decaying_idempotent_visits;
        // ID for the trace. It is unset if replays == 0.
        TraceID tid;

        // visit updates the TraceMeta's decaying visit count when visited
        // at opidx.
        void visit(size_t opidx);
        // score computes the TraceMeta's score when observed at opidx.
        double score(size_t opidx) const;
        // R is the exponential rate of decay for a trace.
        static constexpr double R = 0.99;
        // SCORE_CAP_MULT is the multiplier for how large the score
        // of a particular trace can ever get.
        static constexpr double SCORE_CAP_MULT = 10;
        // REPLAY_SCALE is at most how much a score should be increased
        // to favor replays.
        static constexpr size_t REPLAY_SCALE = 2;
        // IDEMPOTENT_VISIT_SCALE is at most how much a score should
        // be increased to favor idempotent replays.
        static constexpr double IDEMPOTENT_VISIT_SCALE = 2.0;
      };
      Trie<Murmur3Hasher::Hash, TraceMeta> trie;

      // For watching and maintaining decaying visit counts
      // of pointers for scoring.
      class WatchPointer {
      public:
        WatchPointer(TrieNode<Murmur3Hasher::Hash, TraceMeta>* node_, size_t opidx_)
            : node(node_), opidx(opidx_) { }
        // This pointer only has an advance function, as there's nothing
        // to do on commit.
        bool advance(Murmur3Hasher::Hash token);
        size_t get_opidx() const { return this->opidx; }
      private:
        TrieNode<Murmur3Hasher::Hash, TraceMeta>* node;
        size_t opidx;
      };
      std::vector<WatchPointer> active_watching_pointers;

      // For the actual committed trie.
      class CommitPointer {
      public:
        CommitPointer(TrieNode<Murmur3Hasher::Hash, TraceMeta>* node_, size_t opidx_)
          : node(node_), opidx(opidx_), depth(0) { }
        bool advance(Murmur3Hasher::Hash token);
        void advance_for_trace_noop() { this->depth++; }
        bool complete();
        TraceID replay(OperationExecutor* executor);
        double score(size_t opidx);
        size_t get_opidx() const { return this->opidx; }
        size_t get_length() { return this->depth; }
      private:
        TrieNode<Murmur3Hasher::Hash, TraceMeta>* node;
        size_t opidx;
        // depth is the number of operations (traceable and trace no-ops)
        // contained within the trace.
        size_t depth;
      };
      std::vector<CommitPointer> active_commit_pointers;
      std::vector<CommitPointer> completed_commit_pointers;


      // Fields for the management of pending operations.
      std::queue<Operation*> operations;
      size_t operation_start_idx;

      // flush_buffer executes operations until opidx, or flushes
      // the entire operation buffer if no opidx is provided.
      void flush_buffer();
      void flush_buffer(size_t opidx);
      // replay_trace executes operations under the trace tid
      // until opidx, after which it inserts an end trace.
      void replay_trace(size_t opidx, TraceID tid);
    };

    template <typename T>
    class AutomaticTracingContext : public T,
                                    public OperationExecutor,
                                    public TraceProcessingJobExecutor {
    public:
      template <typename ... Args>
      AutomaticTracingContext(Args&& ... args)
        : T(std::forward<Args>(args) ... ),
          opidx(0),
          identifier(this,
                     this->watcher,
                     this->runtime->auto_trace_batchsize,
                     this->runtime->auto_trace_max_start_watch,
                     this->runtime->auto_trace_in_flight_jobs,
                     this->runtime->auto_trace_wait_async_jobs,
                     this->runtime->auto_trace_min_trace_length),
          watcher(this->replayer, this->runtime->auto_trace_commit_threshold),
          replayer(this)
        {
          // Perform any initialization for async trace analysis needed.
          T::initialize_async_trace_analysis(this->runtime->auto_trace_in_flight_jobs);
        }
    public:
      bool add_to_dependence_queue(Operation *op,
                                   bool unordered = false,
                                   bool outermost = true) override;
      // get_new_unique_hash() generates a hash value that
      // will not repeat. This is used to represent operations
      // or events that are not traceable, so that the trace
      // identification analysis does not identify repeats that
      // cross over untraceable operations.
      Murmur3Hasher::Hash get_new_unique_hash();
      // If the application performs a blocking operation, we need to know
      // about that, so override TaskContext::record_blocking_call().
      void record_blocking_call(bool in_operation_stream) override;
    public:
      // Overrides for OperationExecutor.
      TraceID get_fresh_trace_id() override;
      void issue_begin_trace(TraceID id) override;
      void issue_end_trace(TraceID id) override;
      bool issue_operation(Operation* op, bool unordered = false, bool outermost = false) override;
    public:
      // Overrides for TraceJobProcessingExecutor.
      RtEvent enqueue_task(const InnerContext::AutoTraceProcessRepeatsArgs& args, size_t opidx, bool wait) override;
      RtEvent poll_pending_tasks(size_t opidx, bool must_pop) override;
    private:
      size_t opidx;
      BatchedTraceIdentifier identifier;
      TraceOccurrenceWatcher watcher;
      TraceReplayer replayer;
      // Unfortunately, operation context indexes are assigned
      // at issue time in the runtime, which unfortunately happens
      // _before_ operations pass through add_to_dependence_queue, and
      // these ID's are required to be monotonically increasing. This
      // can become a problem for us as we are going to insert some
      // begin and end trace operations out of order with respect to
      // ID assignment in the runtime. So, what we're going to do is
      // maintain our own context index for issued operations (separately
      // from the existing operation index counter), and rewrite the
      // indexes of issued operations on the fly with the new index
      // we are maintaining here.
      size_t rewritten_op_context_idx = 0;
      // We need to maintain whether we are tracing separately from the
      // inner context, as we'll need to manually attach traces to operations.
      // Like above, the trace is set when the operation is created, rather
      // than when it is added to the dependence queue. So we have to set
      // this ourselves before sending into the internal dependence queue.
      bool started_auto_trace = false;
      // unique_hash_idx_counter maintains a counter of non-traceable
      // operations seen so far, used to generate unique hashes for
      // those operations.
      size_t unique_hash_idx_counter = 0;
    };

    void auto_trace_process_repeats(const void* args);

    // Utility functions.
    bool is_operation_traceable(Operation* op);
    bool is_operation_ignorable_in_traces(Operation* op);


    // TODO (rohany): Can we move these declarations to another file?

    template <typename T>
    bool AutomaticTracingContext<T>::add_to_dependence_queue(Operation* op, bool unordered, bool outermost) {
      // If we have an unordered operation, just forward it directly without
      // getting it involved in the tracing infrastructure.
      if (unordered) {
        return this->issue_operation(op, unordered, outermost);
      }

      // TODO (rohany): In the future, if we allow automatic and explicit tracing
      //  in the same program, we're going to need to be able to know whether
      //  trace operations have been issued by the AutomaticTracingContext
      //  or the application. It's a bit of plumbing right now to get this
      //  through to all of the applications, so we'll start with the ones
      //  where it's present, and then disallow trace operations to be issued
      //   by the application.
      // Trace operations that we recognize go straight through the context into
      // the dependence queue, as we're only going to see them as a callback
      // that we issue while we're already in the dependence queue. This because
      // the callback structure is
      // AutoTracingContext::add_to_dependence_queue ->
      // AutoTracingContext::replay_trace ->
      // InnerContext::begin_trace ->
      // AutoTracingContext::add_to_dependence_queue -> HERE.
      switch (op->get_operation_kind()) {
        case Operation::OpKind::TRACE_BEGIN_OP_KIND: // Fallthrough.
        case Operation::OpKind::TRACE_REPLAY_OP_KIND: {
          assert(op->get_trace()->tid >= LEGION_MAX_APPLICATION_TRACE_ID && op->get_trace()->tid < LEGION_INITIAL_LIBRARY_ID_OFFSET);
          return this->issue_operation(op);
        }
        case Operation::OpKind::TRACE_CAPTURE_OP_KIND: // Fallthrough.
        case Operation::OpKind::TRACE_COMPLETE_OP_KIND: {
          return this->issue_operation(op);
        }
        default: {
          break;
        }
      }

      // If we encounter a traceable operation, then it's time to start
      // analyzing it and adding it the corresponding operation processors.
      if (is_operation_traceable(op)) {
        Murmur3Hasher::Hash hash = TraceHashHelper{}.hash(op);
        // TODO (rohany): Have to have a hash value that can be used as the sentinel $
        //  token for the suffix tree processing algorithms.
        assert(!(hash.x == 0 && hash.y == 0));
        this->identifier.process(hash, this->opidx);
        this->watcher.process(hash, this->opidx);
        this->replayer.process(op, hash, this->opidx);
        this->opidx++;
        return true;
      } else if (is_operation_ignorable_in_traces(op)) {
        // If the operation we are processing is "ignorable" in traces
        // then we won't consider it for trace identification or counting
        // watches in the identifier and watcher. The idea here is to thread
        // these operations through the pipeline unless we are replaying a
        // trace, in which case they will be dropped.
        this->replayer.process_trace_noop(op);
        this->opidx++;
        return true;
      } else {
        // When encountering a non-traceable operation, insert a
        // dummy hash value into the trace identifier so that the
        // traces it finds don't span across these operations.
        this->identifier.process(this->get_new_unique_hash(), this->opidx);

        // When encountering a non-traceable operation, invalidate
        // all active pointers from the TraceOccurrenceWatcher, as
        // this operation has broken any active traces.
        this->watcher.clear();

        // If we see a non-traceable operation, then we need to flush
        // all of the pending operations sitting in the replayer (as
        // a trace is no longer possible to replay) before issuing
        // the un-traceable operation.
        this->replayer.flush(this->opidx);
        log_auto_trace.debug() << "Encountered untraceable operation: "
                               << Operation::get_string_rep(op->get_operation_kind());
        return this->issue_operation(op, unordered, outermost);
      }
    }


    template <typename T>
    TraceID AutomaticTracingContext<T>::get_fresh_trace_id() {
      return this->generate_dynamic_trace_id();
    }

    template <typename T>
    void AutomaticTracingContext<T>::issue_begin_trace(Legion::TraceID id) {
      T::begin_trace(
        id,
        false /* logical_only */,
        false /* static_trace */,
        nullptr /* managed */,
        false /* dep */,
        nullptr /* provenance */,
        false /* from application */
      );
      // Set started_auto_trace after issuing the begin trace, as it
      // will set T::current_trace.
      this->started_auto_trace = true;
    }

    template <typename T>
    void AutomaticTracingContext<T>::issue_end_trace(Legion::TraceID id) {
      // Set started_auto_trace to be false before the operation, as the
      // issuing of the end trace will set T::current_trace.
      this->started_auto_trace = false;
      T::end_trace(id, false /* deprecated */, nullptr /* provenance */, false /* from application */);
    }

    template <typename T>
    bool AutomaticTracingContext<T>::issue_operation(Legion::Internal::Operation *op, bool unordered, bool outermost) {
      // Set and then update the separately maintained opidx counter.
      if (!unordered) {
        // If we're tracing and this operation is a trace no-op, then no-op.
        if (this->started_auto_trace && is_operation_ignorable_in_traces(op)) {
          this->rewritten_op_context_idx++;
          return true;
        }

        op->set_ctx_index(this->rewritten_op_context_idx);
        this->rewritten_op_context_idx++;
        // TODO (rohany): This might not be needed once Mike refactors
        //  context index assignment.
        if (this->started_auto_trace) {
          assert(this->current_trace != nullptr);
          op->set_trace(this->current_trace, nullptr /* dependencies */);
        }
      }
      return T::add_to_dependence_queue(op, unordered, outermost);
    }

    template <typename T>
    void AutomaticTracingContext<T>::record_blocking_call(bool in_operation_stream) {
      if (in_operation_stream) {
        // Handling waits from the application is very similar
        // to the case in add_to_dependence_queue when we encounter an
        // operation that is not traceable. We interrupt traces in
        // the identifier, and flush the watcher and replayer.
        this->identifier.process(this->get_new_unique_hash(), this->opidx);
        this->watcher.clear();
        this->replayer.flush(this->opidx);
      }
      // Need to also do whatever the base context was going to do.
      T::record_blocking_call(in_operation_stream);
    }

    template <typename T>
    Murmur3Hasher::Hash AutomaticTracingContext<T>::get_new_unique_hash() {
      size_t idx = this->unique_hash_idx_counter;
      this->unique_hash_idx_counter++;
      Murmur3Hasher hasher(false /* precise */);
      hasher.hash(Operation::OpKind::LAST_OP_KIND);
      hasher.hash(idx);
      return hasher.get_hash();
    }

    template <typename T>
    RtEvent AutomaticTracingContext<T>::enqueue_task(
        const InnerContext::AutoTraceProcessRepeatsArgs& args,
        size_t opidx,
        bool wait
    ) {
      return T::enqueue_trace_analysis_meta_task(args, opidx, wait);
    }

    template <typename T>
    RtEvent AutomaticTracingContext<T>::poll_pending_tasks(
        size_t opidx,
        bool must_pop
    ) {
      return T::poll_pending_trace_analysis_tasks(opidx, must_pop);
    }

    template <typename T>
    void TraceOccurrenceWatcher::insert(T start, T end, size_t opidx) {
      this->trie.insert(start, end, TraceMeta(opidx));
    }

    template <typename T>
    bool TraceOccurrenceWatcher::prefix(T start, T end) {
      return this->trie.prefix(start, end);
    }

    template <typename T>
    void TraceReplayer::insert(T start, T end, size_t opidx) {
      return this->trie.insert(start, end, TraceMeta(opidx, std::distance(start, end)));
    }

    template <typename T>
    bool TraceReplayer::prefix(T start, T end) {
      return this->trie.prefix(start, end);
    }
  };
};

#endif // __LEGION_AUTO_TRACE_H__
