// Copyright 2015 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// test for Realm's serializing code

#include "realm/serialize.h"

#include <sys/resource.h>
#include <string.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <list>

static bool verbose = false;
static int error_count = 0;

static void parse_args(int argc, const char *argv[])
{
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-v")) {
      verbose = true;
      continue;
    }
  }
}

// helper functions to print out containers
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
  os << '[' << v.size() << ']';
  for(typename std::vector<T>::const_iterator it = v.begin();
      it != v.end();
      it++)
    os << ' ' << *it;
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::list<T>& l)
{
  os << '[' << l.size() << ']';
  for(typename std::list<T>::const_iterator it = l.begin();
      it != l.end();
      it++)
    os << ' ' << *it;
  return os;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::map<T1, T2>& m)
{
  os << '[' << m.size() << ']';
  for(typename std::map<T1, T2>::const_iterator it = m.begin();
      it != m.end();
      it++)
    os << ' ' << it->first << '=' << it->second;
  return os;
}

template <typename T>
void test_dynamic(const char *name, const T& input, size_t exp_size = 0)
{
  // first serialize and check size
  Realm::Serialization::DynamicBufferSerializer dbs(0);

  bool ok1 = dbs << input;
  if(!ok1) {
    std::cout << "ERROR: " << name << "dynamic serialization failed!" << std::endl;
    error_count++;
  }

  size_t act_size = dbs.bytes_used();
  
  if(exp_size > 0) {
    if(act_size != exp_size) {
      std::cout << "ERROR: " << name << "dynamic size = " << act_size << " (should be " << exp_size << ")" << std::endl;
      error_count++;
    } else {
      if(verbose)
	printf("OK: %s dynamic size = %zd\n", name, act_size);
    }
  }

  void *buffer = dbs.detach_buffer();

  // now deserialize into a new object and test for equality
  Realm::Serialization::FixedBufferDeserializer fbd(buffer, act_size);
  T output;

  bool ok2 = fbd >> output;
  if(!ok2) {
    printf("ERROR: %s dynamic deserialization failed!\n", name);
    error_count++;
  }

  ptrdiff_t leftover = fbd.bytes_left();
  if(leftover != 0) {
    printf("ERROR: %s dynamic leftover = %zd\n", name, leftover);
    error_count++;
  }

  bool ok3 = (input == output);
  if(ok3) {
    if(verbose)
      printf("OK: %s dynamic output matches\n", name);
  } else {
    printf("ERROR: %s dynamic output mismatch:\n", name);
    std::cout << "Input:  " << input << std::endl;
    std::cout << "Buffer: [" << act_size << "]";
    std::cout << std::hex;
    for(size_t i = 0; i < act_size; i++)
      std::cout << ' ' << std::setfill('0') << std::setw(2) << (int)((unsigned char *)buffer)[i];
    std::cout << std::dec << std::endl;
    std::cout << "Output: " << output << std::endl;
    error_count++;
  }

  free(buffer);
}

template <typename T>
void do_test(const char *name, const T& input, size_t exp_size = 0)
{
  test_dynamic(name, input, exp_size);
}

template <typename T1, typename T2>
struct Pair {
  T1 x;
  T2 y;

  Pair(void) : x(), y() {}
  Pair(T1 _x, T2 _y) : x(_x), y(_y) {}

  bool operator==(const Pair<T1,T2>& rhs) const
  {
    return (x == rhs.x) && (y == rhs.y);
  }

  friend std::ostream& operator<<(std::ostream& os, const Pair<T1, T2>& p)
  {
    return os << '<' << p.x << ',' << p.y << '>';
  }
};

#if 0
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const Pair<T1, T2>& p)
{
  return os << '<' << p.x << ',' << p.y << '>';
}
#endif

typedef Pair<double, int> PODStruct;
TYPE_IS_SERIALIZABLE(PODStruct);

typedef Pair<double, float> PODPacked;
template <typename S>
bool operator<<(S& s, const PODPacked& p) { return (s << p.x) && (s << p.y); }
template <typename S>
bool operator>>(S& s, PODPacked& p) { return (s >> p.x) && (s >> p.y); }

typedef Pair<double, char> PODPacked2;
template <typename S>
bool operator&(S& s, const PODPacked& p) { return (s & p.x) && (s & p.y); }

int main(int argc, const char *argv[])
{
  parse_args(argc, argv);

  {
    // a common failure mode for the serialization logic is infinite recursion,
    //  so set very tight bounds on our stack size and run time
    struct rlimit rl;
    int ret;
    rl.rlim_cur = rl.rlim_max = 16384;  // 16KB
    ret = setrlimit(RLIMIT_STACK, &rl);
    assert(ret == 0);
    rl.rlim_cur = rl.rlim_max = 5;  // 5 seconds
    ret = setrlimit(RLIMIT_CPU, &rl);
    assert(ret == 0);
  }

  int x = 5;
  do_test("int", x, sizeof(int));

  do_test("double", double(4.5), sizeof(double));

  //void *f = &x;
  //do_test("void*", f, sizeof(void *));

  do_test("pod struct", PODStruct(6.5, 7), sizeof(PODStruct));

  do_test("pod packed", PODPacked(8.5, 9.1), 12 /* not sizeof(PODPacked)*/);

  //do_test("pod packed2", PODPacked2(10.5, 'z'), 9 /* not sizeof(PODPacked2)*/);

  std::vector<int> a(3);
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  do_test("vector<int>", a, sizeof(size_t) + a.size() * sizeof(int));

  std::list<int> b;
  b.push_back(4);
  b.push_back(5);
  b.push_back(6);
  b.push_back(7);
  do_test("list<int>", b, sizeof(size_t) + b.size() * sizeof(int));

  std::map<int, double> c;
  c[8] = 1.1;
  c[9] = 2.2;
  c[10] = 3.3;
  do_test("map<int,double>", c, sizeof(size_t) + c.size() * 16 /*alignment*/);
}
