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

#include "realm/operation.h"

#include "realm/faults.h"
#include "realm/runtime_impl.h"

namespace Realm {

  Logger log_optable("optable");


  ////////////////////////////////////////////////////////////////////////
  //
  // class Operation
  //

  Operation::~Operation(void)
  {
    // delete all of the async work items we were given to track
    for(std::set<AsyncWorkItem *>::iterator it = all_work_items.begin();
	it != all_work_items.end();
	it++)
      delete *it;
    all_work_items.clear();
  }

  bool Operation::mark_ready(void)
  {
    // attempt to switch from WAITING -> READY
    Status::Result prev = __sync_val_compare_and_swap(&status.result,
						      Status::WAITING,
						      Status::READY);
    switch(prev) {
    case Status::WAITING:
      {
	// normal behavior
	timeline.record_ready_time();
	return true;
      }

    case Status::CANCELLED:
      {
	// lost the race to a cancellation request
	return false;
      }

    default:
      {
	assert(0 && "mark_ready called when not WAITING or CANCELLED");
      }
    }
    return false;
  }

  bool Operation::mark_started(void)
  {
    // attempt to switch from READY -> RUNNING
    Status::Result prev = __sync_val_compare_and_swap(&status.result,
						      Status::READY,
						      Status::RUNNING);
    switch(prev) {
    case Status::READY:
      {
	// normal behavior
	timeline.record_start_time();
	return true;
      }

    case Status::CANCELLED:
      {
	// lost the race to a cancellation request
	return false;
      }

    default:
      {
	assert(0 && "mark_started called when not READY or CANCELLED");
      }
    }
    return false;
  }

  void Operation::mark_finished(bool successful)
  {
    timeline.record_end_time();

    // update this count first
    if(!successful)
      __sync_fetch_and_add(&failed_work_items, 1);

    // do an atomic decrement of the work counter to see if we're also complete
    int remaining = __sync_sub_and_fetch(&pending_work_items, 1);

    if(remaining == 0)
      mark_completed();    
  }

  void Operation::mark_terminated(int error_code, const ByteArray& details)
  {
    // attempt to switch from RUNNING -> TERMINATED_EARLY
    Status::Result prev = __sync_val_compare_and_swap(&status.result,
						      Status::RUNNING,
						      Status::TERMINATED_EARLY);
    if(prev == Status::RUNNING) {
      status.error_code = error_code;
      status.error_details = details;
    } else {
      // if that didn't work, try going from INTERRUPT_REQUESTED -> TERMINATED_EARLY
#ifndef NDEBUG
      prev =
#endif
        __sync_val_compare_and_swap(&status.result,
				    Status::INTERRUPT_REQUESTED,
				    Status::TERMINATED_EARLY);
      assert(prev == Status::INTERRUPT_REQUESTED);
      // don't update error_code/details - that was already provided in the interrupt request
    }

    timeline.record_complete_time();

    __sync_fetch_and_add(&failed_work_items, 1);

    // if this operation has async work items, try to cancel them
    if(!all_work_items.empty()) {
      for(std::set<AsyncWorkItem *>::iterator it = all_work_items.begin();
	  it != all_work_items.end();
	  it++)
	(*it)->request_cancellation();
    }

    // can't trigger the finish event immediately if async work items are pending
    int remaining = __sync_sub_and_fetch(&pending_work_items, 1);

    if(remaining == 0)
      mark_completed();    
  }

  void Operation::mark_completed(void)
  {
    // don't overwrite a TERMINATED_EARLY or CANCELLED status
    Status::Result newresult = ((failed_work_items == 0) ?
  				  Status::COMPLETED_SUCCESSFULLY :
				  Status::COMPLETED_WITH_ERRORS);
#ifndef NDEBUG
    Status::Result prev =
#endif
      __sync_val_compare_and_swap(&status.result,
				  Status::RUNNING,
				  newresult);
    assert((prev == Status::RUNNING) ||
	   (prev == Status::TERMINATED_EARLY) ||
	   (prev == Status::CANCELLED));

    timeline.record_complete_time();

    send_profiling_data();

    // trigger the finish event last - the OperationTable will delete us shortly after we do
    // poison if there were any failed work items
    trigger_finish_event(failed_work_items != 0);
  }

  bool Operation::attempt_cancellation(int error_code,
				       const void *reason_data, size_t reason_size)
  {
    // all we know how to do here is convert from WAITING or READY to CANCELLED
    // there's no mutex, so we'll attempt to update the status with a sequence of 
    //  compare_and_swap's, making sure to follow the normal progression of status updates
    if(__sync_bool_compare_and_swap(&status.result, Status::WAITING, Status::CANCELLED) ||
       __sync_bool_compare_and_swap(&status.result, Status::READY, Status::CANCELLED)) {
      status.error_code = error_code;
      status.error_details.set(reason_data, reason_size);

      // we can't call mark_finished here, because we don't know if the caller owns the 
      //  reference that will be released by that call, so let the caller do that (or
      //  the owner can do it when they call mark_ready/started and get a failure code)

      return true;
    }

    // if the task is in a terminal state, no subclass will be able to do anything either
    if((status.result == Status::COMPLETED_SUCCESSFULLY) ||
       (status.result == Status::COMPLETED_WITH_ERRORS) ||
       (status.result == Status::TERMINATED_EARLY) ||
       (status.result == Status::CANCELLED))
      return true;

    // otherwise we return false - a subclass might override and add additional ways to
    //  cancel an already-running operation
    return false;
  }

  void Operation::set_priority(int new_priority)
  {
    // ignored
  }

  // a common reason for cancellation is a poisoned precondition - this helper takes care
  //  of recording the error code and marking the operation as (unsuccessfully) finished
  void Operation::handle_poisoned_precondition(Event pre)
  {
    // there should be no race conditions for this - state should be WAITING because we
    //  know there's a precondition that didn't successfully trigger
    assert(status.result == Status::WAITING);
    status.error_code = Faults::ERROR_POISONED_PRECONDITION;
    status.error_details.set(&pre, sizeof(pre));

    status.result = Status::CANCELLED;
    mark_finished(false /*unsuccessful*/);
  }

  void Operation::send_profiling_data(void)
  {
    if(requests.request_count() > 0) {
      if(measurements.wants_measurement<ProfilingMeasurements::OperationStatus>())
	measurements.add_measurement(status);

      if(measurements.wants_measurement<ProfilingMeasurements::OperationTimeline>())
	measurements.add_measurement(timeline);

      if(measurements.wants_measurement<ProfilingMeasurements::OperationEventWaits>())
	measurements.add_measurement(waits);

      measurements.send_responses(requests);
    }
  }

  void Operation::trigger_finish_event(bool poisoned)
  {
    if(finish_event.exists())
      GenEventImpl::trigger(finish_event, poisoned);
#ifndef REALM_USE_OPERATION_TABLE
    // no operation table to decrement the refcount, so do it ourselves
    // SJT: should this always be done for operations without finish events?
    remove_reference();
#endif
  }

  void Operation::clear_profiling(void)
  {
    requests.clear();
    measurements.clear();
  }

  void Operation::reconstruct_measurements()
  {
    measurements.import_requests(requests);
    timeline.record_create_time();
  }

  std::ostream& operator<<(std::ostream& os, const Operation *op)
  {
    op->print(os);
    os << " status=" << op->status.result
       << "(" << op->timeline.ready_time
       << "," << op->timeline.start_time
       << ") work=" << op->pending_work_items;
    if(!op->all_work_items.empty()) {
      os << " { ";
      std::set<Operation::AsyncWorkItem *>::const_iterator it = op->all_work_items.begin();
      (*it)->print(os);
      while(++it != op->all_work_items.end()) {
	os << ", ";
	(*it)->print(os);
      }
      os << " }\n";
    }
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class OperationTable::TableEntry
  //

#if 0
  OperationTable::TableEntry::TableEntry(Event _finish_event)
    : finish_event(_finish_event)
    , local_op(0)
    , remote_node(-1)
    , pending_cancellation(false)
  {}
#endif

  ////////////////////////////////////////////////////////////////////////
  //
  // class OperationTable::TableCleaner
  //

  OperationTable::TableCleaner::TableCleaner(OperationTable *_table)
    : table(_table)
  {}

  bool OperationTable::TableCleaner::event_triggered(Event e, bool poisoned)
  {
    table->event_triggered(e);
    return false;  // never delete us
  }

  void OperationTable::TableCleaner::print(std::ostream& os) const
  {
    os << "operation table cleaner (table=" << table << ")";
  }

  Event OperationTable::TableCleaner::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class OperationTable
  //

  OperationTable::OperationTable(void)
#ifdef REALM_USE_OPERATION_TABLE
    : cleaner(this)
#endif
  {}

  OperationTable::~OperationTable(void)
  {}

  // Operations are 'owned' by the table - the table will free them once it
  //  gets the completion event for it
  void OperationTable::add_local_operation(Event finish_event, Operation *local_op)
  {
#ifdef REALM_USE_OPERATION_TABLE
    // cast local_op to void * to avoid pretty-printing
    log_optable.info() << "event " << finish_event << " added: local_op=" << (void *)local_op;

    // "hash" the id to figure out which subtable to use
    int subtable = finish_event.id % NUM_TABLES;
    GASNetHSL& mutex = mutexes[subtable];
    Table& table = tables[subtable];

    bool cancel_immediately = false;
    void *reason_data = 0;
    size_t reason_size = 0;
    {
      AutoHSLLock al(mutex);

      // see if we have any info for this event?
      Table::iterator it = table.find(finish_event);
      if(it == table.end()) {
	// new entry - create one and it inherits the refcount
	TableEntry& e = table[finish_event];
	e.local_op = local_op;
	e.remote_node = -1;
	e.pending_cancellation = false;
	e.reason_data = 0;
	e.reason_size = 0;
      } else {
	// existing entry should only occur if there's a pending cancellation
	TableEntry& e = it->second;
	assert(e.local_op == 0);
	assert(e.remote_node == -1);
	assert(e.pending_cancellation);

	// put the operation in the table in case anybody else comes along while we're trying to
	//  cancel it - add a reference since we're keeping one
	e.local_op = local_op;
	local_op->add_reference();
	cancel_immediately = true;
	reason_data = e.reason_data;
	reason_size = e.reason_size;
      }
    }

    // either way there's an entry in the table for this now, so make sure our cleaner knows
    //  to clean it up
    EventImpl::add_waiter(finish_event, &cleaner);

    // and finally, perform a delayed cancellation if requested
    if(cancel_immediately) {
      bool did_cancel = local_op->attempt_cancellation(Realm::Faults::ERROR_CANCELLED,
						       reason_data, reason_size);
      if(reason_data)
	free(reason_data);
      log_optable.info() << "event " << finish_event << " - operation " << (void *)local_op << " cancelled=" << did_cancel;
      local_op->remove_reference();
    }
#endif
  }

  void OperationTable::add_remote_operation(Event finish_event, int remote_node)
  {
#ifdef REALM_USE_OPERATION_TABLE
    log_optable.info() << "event " << finish_event << " added: remote_node=" << remote_node;

    // "hash" the id to figure out which subtable to use
    int subtable = finish_event.id % NUM_TABLES;
    GASNetHSL& mutex = mutexes[subtable];
    Table& table = tables[subtable];

    {
      AutoHSLLock al(mutex);

      // no duplicates allowed here - a local cancellation request cannot occur until we
      //  return
      assert(table.find(finish_event) == table.end());

      TableEntry& e = table[finish_event];
      e.local_op = 0;
      e.remote_node = remote_node;
      e.pending_cancellation = false;
      e.reason_data = 0;
      e.reason_size = 0;
    }

    // we can remove this entry once we know the operation is complete
    EventImpl::add_waiter(finish_event, &cleaner);
#endif
  }

  void OperationTable::event_triggered(Event finish_event)
  {
#ifdef REALM_USE_OPERATION_TABLE
    // "hash" the id to figure out which subtable to use
    int subtable = finish_event.id % NUM_TABLES;
    GASNetHSL& mutex = mutexes[subtable];
    Table& table = tables[subtable];

    Operation *local_op = 0;
    {
      AutoHSLLock al(mutex);

      // get the entry - it must exist
      Table::iterator it = table.find(finish_event);
      assert(it != table.end());

      // if there was a local op, remember it so we can remove the reference outside of this mutex
      local_op = it->second.local_op;

      table.erase(it);
    }

    log_optable.info() << "event " << finish_event << " cleaned: local_op=" << (void *)local_op;

    if(local_op)
      local_op->remove_reference();
#else
    assert(0);
#endif
  }
    
  void OperationTable::request_cancellation(Event finish_event,
					    const void *reason_data, size_t reason_size)
  {
#ifdef REALM_USE_OPERATION_TABLE
    // "hash" the id to figure out which subtable to use
    int subtable = finish_event.id % NUM_TABLES;
    GASNetHSL& mutex = mutexes[subtable];
    Table& table = tables[subtable];

    bool found = false;
    Operation *local_op = 0;
    int remote_node = -1;
    {
      AutoHSLLock al(mutex);

      Table::iterator it = table.find(finish_event);

      if(it != table.end()) {
	found = true;

	// if there's a local op, we need to take a reference in case it completes successfully
	//  before we get to it below
	if(it->second.local_op) {
	  local_op = it->second.local_op;
	  local_op->add_reference();
	}
	remote_node = it->second.remote_node;
	assert(!it->second.pending_cancellation);
      }
    }

    if(!found) {
      // not found - who owns this event?
      int owner = ID(finish_event).event.creator_node;

      if(owner == my_node_id) {
	// if we're the owner, it's probably for an event that already completed successfully,
	//  so ignore the request
	log_optable.info() << "event " << finish_event << " cancellation ignored - not in table";
      } else {
	// let the owner of the event deal with it
	remote_node = owner;
      }
    }

    if(remote_node != -1) {
      // TODO: active message
      assert(false);
    }

    if(local_op) {
      bool did_cancel = local_op->attempt_cancellation(Realm::Faults::ERROR_CANCELLED,
						       reason_data, reason_size);
      log_optable.info() << "event " << finish_event << " - operation " << (void *)local_op << " cancelled=" << did_cancel;
      local_op->remove_reference();
    }
#else
    assert(0);
#endif
  }

  void OperationTable::set_priority(Event finish_event, int new_priority)
  {
#ifdef REALM_USE_OPERATION_TABLE
    // "hash" the id to figure out which subtable to use
    int subtable = finish_event.id % NUM_TABLES;
    GASNetHSL& mutex = mutexes[subtable];
    Table& table = tables[subtable];

    bool found = false;
    Operation *local_op = 0;
    int remote_node = -1;
    {
      AutoHSLLock al(mutex);

      Table::iterator it = table.find(finish_event);

      if(it != table.end()) {
	found = true;

	// if there's a local op, we need to take a reference in case it completes successfully
	//  before we get to it below
	if(it->second.local_op) {
	  local_op = it->second.local_op;
	  local_op->add_reference();
	}
	remote_node = it->second.remote_node;
      }
    }

    if(!found) {
      // not found - who owns this event?
      int owner = ID(finish_event).event.creator_node;

      if(owner == my_node_id) {
	// if we're the owner, it's probably for an event that already completed successfully,
	//  so ignore the request
	log_optable.info() << "event " << finish_event << " priority change ignored - not in table";
      } else {
	// let the owner of the event deal with it
	remote_node = owner;
      }
    }

    if(remote_node != -1) {
      // TODO: active message
      assert(false);
    }

    if(local_op) {
      local_op->set_priority(new_priority);
      log_optable.info() << "event " << finish_event << " - operation " << (void *)local_op << " priority=" << new_priority;
      local_op->remove_reference();
    }
#else
    assert(0);
#endif
  }
    
  /*static*/ void OperationTable::register_handlers(void)
  {
  }

  void OperationTable::print_operations(std::ostream& os)
  {
#ifdef REALM_USE_OPERATION_TABLE
    os << "OperationTable(node=" << my_node_id << ") {\n";

    for(int subtable = 0; subtable < NUM_TABLES; subtable++) {
      GASNetHSL& mutex = mutexes[subtable];
      Table& table = tables[subtable];

      // taking the lock on the subtable also guarantees that none of the
      //  operations in the subtable will be deleted during our iteration
      AutoHSLLock al(mutex);

      for(Table::const_iterator it = table.begin();
	  it != table.end();
	  ++it) {
	if(it->second.local_op) {
	  os << "  " << it->first << ": " << it->second.local_op << "\n";
	} else {
	  os << "  " << it->first << ": remote - node=" << it->second.remote_node << "\n";
	}
      }
    }

    os << "}\n";
#endif
  }


}; // namespace Realm
