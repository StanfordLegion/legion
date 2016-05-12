/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <vector>

class Event {
 public:
  static Event *get_event(unsigned long long ev_id, unsigned gen);
  static void link_events(Event *p, Event *s);

  static int count_stalled(void);
  static void find_loops(void);

  void set_desc(const char *s);
  void add_waiting_thread(unsigned long long thr_id);

  unsigned long long id;
  unsigned gen;
  std::set<Event *> preds, succs;
  std::vector<unsigned long long> threads;
  char *desc;
  bool visited;

  static std::map<unsigned long long, std::map<unsigned, Event *> *> events;

 protected:
  Event(unsigned long long _id, unsigned _gen);
};

std::map<unsigned long long, std::map<unsigned, Event *> *> Event::events;

Event::Event(unsigned long long _id, unsigned _gen)
  : id(_id), gen(_gen), desc(0) {}

/*static*/ Event *Event::get_event(unsigned long long ev_id, unsigned gen)
{
  std::map<unsigned, Event *> *gens;
  std::map<unsigned long long, std::map<unsigned, Event *> *>::iterator it = events.find(ev_id);
  if(it == events.end()) {
    gens = new std::map<unsigned, Event *>;
    events[ev_id] = gens;
  } else {
    gens = it->second;
  }

  std::map<unsigned, Event *>::iterator it2 = gens->find(gen);
  if(it2 == gens->end()) {
    Event *e = new Event(ev_id, gen);
    (*gens)[gen] = e;
    return e;
  } else {
    return it2->second;
  }
}

/*static*/ void Event::link_events(Event *p, Event *s)
{
  //printf("%x/%d -> %x/%d\n", p->id, p->gen, s->id, s->gen);
  assert(p->succs.find(s) == p->succs.end());
  assert(s->preds.find(p) == s->preds.end());

  p->succs.insert(s);
  s->preds.insert(p);
}

void Event::set_desc(const char *s)
{
  if(desc != 0) {
    assert(!strcmp(desc, s));
    return;
  }
  desc = strdup(s);
}

void Event::add_waiting_thread(unsigned long long thr_id)
{
  threads.push_back(thr_id);
}

/*static*/ int Event::count_stalled(void)
{
  int count = 0;
  for(std::map<unsigned long long, std::map<unsigned, Event *> *>::iterator it = events.begin();
      it != events.end();
      it++) {
    for(std::map<unsigned, Event *>::iterator it2 = it->second->begin();
        it2 != it->second->end();
        it2++) {
      Event *e = it2->second;
      if(e->preds.empty()) {
        //printf("Event with no preds: %x/%d: %s\n", e->id, e->gen, e->desc);
        count++;
      }
    }
  }
  return count;
}

static void walk_tree(Event *e, std::vector<Event *>& stack)
{
  if(e->visited) return;

  for(std::vector<Event *>::iterator it = stack.begin(); it != stack.end(); it++)
    if(*it == e) {
      // LOOP!
      printf("LOOP DETECTED!\n");
      std::vector<Event *>::reverse_iterator it = stack.rbegin();
      do {
        printf("  %llx/%d: (%3zd/%3zd) %s", 
               (*it)->id, (*it)->gen, (*it)->preds.size(), (*it)->succs.size(), (*it)->desc);
        if(*it == e) return;
        it++;
      } while(1);
      e->visited = true;
      return;
    }

  // if we're here, no loops, so put ourselves on the stack and keep searching
  stack.push_back(e);

  for(std::set<Event *>::iterator it = e->preds.begin();
      it != e->preds.end();
      it++)
    walk_tree(*it, stack);

  // done, now take ourselves off the stack and mark ourselves visited
  stack.pop_back();
  e->visited = true;
}

/*static*/ void Event::find_loops(void)
{
  for(std::map<unsigned long long, std::map<unsigned, Event *> *>::iterator it = events.begin();
      it != events.end();
      it++) {
    for(std::map<unsigned, Event *>::iterator it2 = it->second->begin();
        it2 != it->second->end();
        it2++) {
      it2->second->visited = false;
    }
  }
  for(std::map<unsigned long long, std::map<unsigned, Event *> *>::iterator it = events.begin();
      it != events.end();
      it++) {
    for(std::map<unsigned, Event *>::iterator it2 = it->second->begin();
        it2 != it->second->end();
        it2++) {
      std::vector<Event *> stack;
      walk_tree(it2->second, stack);
    }
  }
}

int read_events(FILE *f)
{
  char line[256];
  // skip to beginning of events section
  do {
    char *s = fgets(line, 255, f);
    if(!s) {
      printf("WARNING: Unexpected EOF\n");
      return 0;
    }
  } while(strcmp(line, "PRINTING ALL PENDING EVENTS:\n"));

  // now read events until we get to "DONE"
  char *s = fgets(line, 255, f);
  if(!s) {
    printf("WARNING: Unexpected EOF\n");
    return 0;
  }
  while(strcmp(line, "DONE\n")) {
    // Looking for something like this:
    // Event 20000003: gen=110 subscr=0 local=1 remote=1
    unsigned long long ev_id;
    unsigned gen, subscr, nlocal, nfuture, nremote;
    int ret = sscanf(s, "Barrier %llx: gen=%d subscr=%d", 
                     &ev_id, &gen, &subscr);
    if (ret != 3) {
      ret = sscanf(s, "Event %llx: gen=%d subscr=%d local=%d+%d remote=%d",
                   &ev_id, &gen, &subscr, &nlocal, &nfuture, &nremote);
      assert(ret == 6);
    }

    // now read lines that start with a space
    while(1) {
      s = fgets(line, 255, f);
      if(!s) {
        printf("WARNING: Unexpected EOF\n");
        return 0;
      }
      //printf("(%s)\n", s);
      if(s[0] != ' ') break;

      {
        unsigned long long ev2_id;
        unsigned wgen, pos, proc_id, gen2;
        int ret = sscanf(s, "  [%d] L:%*p %nutility thread for processor %x: after=%llx/%d",
                         &wgen, &pos, &proc_id, &ev2_id, &gen2);
        if(ret == 4) {
          Event *e1 = Event::get_event(ev_id, wgen);
          Event *e2 = Event::get_event(ev2_id, gen2);
          Event::link_events(e1, e2);
          e2->set_desc(s + pos);
          continue;
        }
      }

      {
        unsigned long long ev2_id;
        unsigned wgen, pos, gen2;
	int left;
        int ret = sscanf(s, "  [%d] L:%*p - %nevent merger: %llx/%d left=%d",
                         &wgen, &pos, &ev2_id, &gen2, &left);
        if(ret == 4) {
          Event *e1 = Event::get_event(ev_id, wgen);
          Event *e2 = Event::get_event(ev2_id, gen2);
          Event::link_events(e1, e2);
          e2->set_desc(s + pos);
          continue;
        }
      }

      {
        unsigned long long ev2_id;
        unsigned wgen, pos, gen2;
        int ret = sscanf(s, "  [%d] L:%*p - %ndeferred trigger: after=%llx/%d",
                         &wgen, &pos, &ev2_id, &gen2);
        if(ret == 3) {
          Event *e1 = Event::get_event(ev_id, wgen);
          Event *e2 = Event::get_event(ev2_id, gen2);
          Event::link_events(e1, e2);
          e2->set_desc(s + pos);
          continue;
        }
      }

      {
        unsigned long long ev2_id;
        unsigned wgen, pos, gen2;
        int ret = sscanf(s, "  [%d] L:%*p %nGPU Task: %*p after=%llx/%d",
                         &wgen, &pos, &ev2_id, &gen2);
        if(ret == 3) {
          Event *e1 = Event::get_event(ev_id, wgen);
          Event *e2 = Event::get_event(ev2_id, gen2);
          Event::link_events(e1, e2);
          e2->set_desc(s + pos);
          continue;
        }
      }

      {
        unsigned long long ev2_id;
        unsigned wgen, pos, gen2;
        int ret = sscanf(s, "  [%d] L:%*p - %ndma request %*p: after %llx/%d",
                         &wgen, &pos, &ev2_id, &gen2);
        if(ret == 3) {
          Event *e1 = Event::get_event(ev_id, wgen);
          Event *e2 = Event::get_event(ev2_id, gen2);
          Event::link_events(e1, e2);
          e2->set_desc(s + pos);
          continue;
        }
      }

      {
        unsigned long long ev2_id;
        unsigned wgen, pos, gen2;
        int ret = sscanf(s, "  [%d] L:%*p - %ndeferred task: func=%*d proc=%*x finish=%llx/%d",
                         &wgen, &pos, &ev2_id, &gen2);
        if(ret == 3) {
          Event *e1 = Event::get_event(ev_id, wgen);
          Event *e2 = Event::get_event(ev2_id, gen2);
          Event::link_events(e1, e2);
          e2->set_desc(s + pos);
          continue;
        }
      }

      {
        unsigned long long ev2_id;
        unsigned wgen, pos, gen2, ts;
        int delta;
        int ret = sscanf(s, "  [%d] L:%*p %ndeferred arrival: barrier=%llx/%d (%d), delta=%d",
                         &wgen, &pos, &ev2_id, &gen2, &ts, &delta);
        if(ret == 5) {
          Event *e1 = Event::get_event(ev_id, wgen);
          Event *e2 = Event::get_event(ev2_id, gen2);
          Event::link_events(e1, e2);
          e2->set_desc(s + pos);
          continue;
        }
      }

      {
        unsigned long long ev2_id;
        unsigned wgen, gen2;
        unsigned long long thr_id;
        int ret = sscanf(s, "  [%d] L:%*p thread %llx waiting on %llx/%d",
                         &wgen, &thr_id, &ev2_id, &gen2);
        if(ret == 4) {
          Event *e1 = Event::get_event(ev_id, wgen);
          e1->add_waiting_thread(thr_id);
          continue;
        }
      }

      {
	unsigned wgen;
	char dummy;
	unsigned long long thr_id = -1; // unknown
        int ret = sscanf(s, "  [%d] L:%*p - EventTriggeredCondition (thread unknown%c",
                         &wgen, &dummy);
        if(ret == 2) {
          Event *e1 = Event::get_event(ev_id, wgen);
          e1->add_waiting_thread(thr_id);
          continue;
        }
      }

      {
        unsigned wgen;
        unsigned long long thr_id;
        int ret = sscanf(s, "  [%d] L:%*p Waiting greenlet %llx of processor local worker",
                         &wgen, &thr_id);
        if(ret == 2) {
          continue;
        }
      }

      {
        unsigned wgen;
        char dummy;
        int ret = sscanf(s, "  [%d] L:%*p external waiter%c",
                         &wgen, &dummy);
        if(ret == 2) {
          Event *e1 = Event::get_event(ev_id, wgen);
          e1->add_waiting_thread(0);
          continue;
        }
      }

      {
	unsigned wgen;
	void *dummy;
	int ret = sscanf(s, "  [%d] L:%*p - operation table cleaner (table=%p)",
			 &wgen, &dummy);
	if(ret == 2) {
	  // these are probably fine to ignore
	  continue;
	}
      }

      {
        unsigned wgen, dummy;
        int ret = sscanf(s, "  [%d] R: %d",
                         &wgen, &dummy);
        if(ret == 2) {
          // do something with this?
          continue;
        }
      }

      // if we get here, we failed to parse the dependent event
      printf("(%s)\n", s);
      exit(1);
    }
  }
}

int main(int argc, const char *argv[])
{
  for(int i = 1; i < argc; i++) {
    printf("%s\n", argv[i]);
    FILE *f = fopen(argv[i], "r");
    assert(f);
    read_events(f);
    fclose(f);
  }

  int stalled = Event::count_stalled();
  printf("%d stalled events\n", stalled);

  Event::find_loops();
}

