// Copyright 2023 Stanford University
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

#include "realm.h"
#include "realm/serialize.h"

#include <string.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <list>
#include <set>

#include "osdep.h"

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

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& s)
{
  os << '[' << s.size() << ']';
  for(typename std::set<T>::const_iterator it = s.begin();
      it != s.end();
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
size_t test_dynamic(const char *name, const T &input, T &output, size_t exp_size = 0)
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
	std::cout << "OK: " << name << " dynamic size = " << act_size << std::endl;
    }
  }

  void *buffer = dbs.detach_buffer();

  // now deserialize into a new object and test for equality
  Realm::Serialization::FixedBufferDeserializer fbd(buffer, act_size);

  bool ok2 = fbd >> output;
  if(!ok2) {
    std::cout << "ERROR: " << name << " dynamic deserialization failed!" << std::endl;
    error_count++;
  }

  ptrdiff_t leftover = fbd.bytes_left();
  if(leftover != 0) {
    std::cout << "ERROR: " << name << " dynamic leftover = " << leftover << std::endl;
    error_count++;
  }

  bool ok3 = (input == output);
  if(ok3) {
    if(verbose)
      std::cout << "OK: " << name << " dynamic output matches" << std::endl;
  } else {
    std::cout << "ERROR: " << name << " dynamic output mismatch:" << std::endl;
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

  return act_size;
}

template <typename T>
void test_size(const char *name, const T& input, size_t exp_size)
{
  Realm::Serialization::ByteCountSerializer bcs;

  bool ok = bcs << input;
  if(!ok) {
    std::cout << "ERROR: " << name << "byetcount serialization failed!" << std::endl;
    error_count++;
  }

  size_t act_size = bcs.bytes_used();
  if(act_size != exp_size) {
    std::cout << "ERROR: " << name << "bytecount size = " << act_size << " (should be " << exp_size << ")" << std::endl;
    error_count++;
  } else {
    if(verbose)
      std::cout << "OK: " << name << " bytecount size = " << act_size << std::endl;
  }
}

template <typename T>
void test_fixed(const char *name, const T &input, T &output, size_t exp_size)
{
  // first serialize and check size
  void *buffer = malloc(exp_size);
  Realm::Serialization::FixedBufferSerializer fbs(buffer, exp_size);

  bool ok1 = fbs << input;
  if(!ok1) {
    std::cout << "ERROR: " << name << "fixed serialization failed!" << std::endl;
    error_count++;
  }

  ptrdiff_t leftover = fbs.bytes_left();

  if(leftover != 0) {
    std::cout << "ERROR: " << name << "fixed leftover = " << leftover << std::endl;
    error_count++;
  } else {
    if(verbose)
      std::cout << "OK: " << name << " fixed leftover = " << leftover << std::endl;
  }

  // now deserialize into a new object and test for equality
  Realm::Serialization::FixedBufferDeserializer fbd(buffer, exp_size);

  bool ok2 = fbd >> output;
  if(!ok2) {
    std::cout << "ERROR: " << name << " fixed deserialization failed!" << std::endl;
    error_count++;
  }

  leftover = fbd.bytes_left();
  if(leftover != 0) {
    std::cout << "ERROR: " << name << " fixed leftover = " << leftover << std::endl;
    error_count++;
  }

  bool ok3 = (input == output);
  if(ok3) {
    if(verbose)
      std::cout << "OK: " << name << " fixed output matches" << std::endl;
  } else {
    std::cout << "ERROR: " << name << " fixed output mismatch:" << std::endl;
    std::cout << "Input:  " << input << std::endl;
    std::cout << "Buffer: [" << exp_size << "]";
    std::cout << std::hex;
    for(size_t i = 0; i < exp_size; i++)
      std::cout << ' ' << std::setfill('0') << std::setw(2) << (int)((unsigned char *)buffer)[i];
    std::cout << std::dec << std::endl;
    std::cout << "Output: " << output << std::endl;
    error_count++;
  }

  free(buffer);
}

template <typename T>
void do_test(const char *name, const T &input, T &output, size_t exp_size = 0)
{
  exp_size = test_dynamic(name, input, output, exp_size);
  test_size(name, input, exp_size);
  test_fixed(name, input, output, exp_size);
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
bool serialize(S& s, const PODPacked& p) { return (s << p.x) && (s << p.y); }
template <typename S>
bool deserialize(S& s, PODPacked& p) { return (s >> p.x) && (s >> p.y); }

typedef Pair<double, char> PODPacked2;
template <typename S>
bool serdez(S& s, const PODPacked2& p) { return (s & p.x) && (s & p.y); }

typedef Pair<PODPacked, int> PP2;
template <typename S>
bool serdez(S& s, const PP2& p) { return (s & p.x) && (s & p.y); }

int main(int argc, const char *argv[])
{
  parse_args(argc, argv);

#ifndef _MSC_VER
  {
    // a common failure mode for the serialization logic is infinite recursion,
    //  so set very tight bounds on our stack size and run time
    struct rlimit rl;
    int ret;
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
    // macOS on ARM requires higher limit
    rl.rlim_cur = rl.rlim_max = 131072;  // 128KB
#else
    rl.rlim_cur = rl.rlim_max = 16384;  // 16KB
#endif
    ret = setrlimit(RLIMIT_STACK, &rl);
    assert(ret == 0);
    rl.rlim_cur = rl.rlim_max = 5;  // 5 seconds
    ret = setrlimit(RLIMIT_CPU, &rl);
    assert(ret == 0);
  }
#endif

  int x = 5;
  int int_output = 0;
  do_test("int", x, int_output, sizeof(int));

  double double_output = 0.0;
  do_test("double", double(4.5), double_output, sizeof(double));

  //void *f = &x;
  //do_test("void*", f, sizeof(void *));

  PODStruct PODStruct_output(0.0, 0);
  do_test("pod struct", PODStruct(6.5, 7), PODStruct_output, sizeof(PODStruct));

  PODPacked PODPacked_output(0.0, 0.0f);
  do_test("pod packed", PODPacked(8.5, 9.1f), PODPacked_output,
          12 /* not sizeof(PODPacked)*/);

  PODPacked2 PODPacked2_output(0.0, 'z');
  do_test("pod packed2", PODPacked2(10.5, 'z'), PODPacked2_output,
          9 /* not sizeof(PODPacked2)*/);

  PP2 PP2_output = PP2(PODPacked(0.0, 0.0f), 0);
  do_test("pp2", PP2(PODPacked(44.3, 1), 9), PP2_output, 16 /* not sizeof(PP2) */);

  std::vector<int> a(3);
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  std::vector<int> vector_int_output;
  do_test("vector<int>", a, vector_int_output, sizeof(size_t) + a.size() * sizeof(int));

  std::vector<PODStruct> a2(1);
  a2[0] = PODStruct(3, 4);
  std::vector<PODStruct> vector_PODStruct_output;
  do_test("vector<PODStruct>", a2, vector_PODStruct_output,
          (std::max(sizeof(size_t), REALM_ALIGNOF(PODStruct)) +
           a2.size() * sizeof(PODStruct)));

  std::vector<PODPacked> a3(1);
  a3[0] = PODPacked(3, 4);
  std::vector<PODPacked> vector_PODPacked_output;
  do_test(
      "vector<PODPacked>", a3, vector_PODPacked_output,
      (std::max(sizeof(size_t), REALM_ALIGNOF(double)) + 12 /* not sizeof(PODPacked)*/));

  std::vector<PODPacked2> a4(1);
  a4[0] = PODPacked2(3, 4);
  std::vector<PODPacked2> vector_PODPacked2_output;
  do_test(
      "vector<PODPacked2>", a4, vector_PODPacked2_output,
      (std::max(sizeof(size_t), REALM_ALIGNOF(double)) + 9 /* not sizeof(PODPacked2)*/));

  std::list<int> b;
  b.push_back(4);
  b.push_back(5);
  b.push_back(6);
  b.push_back(7);
  std::list<int> list_int_output;
  do_test("list<int>", b, list_int_output, sizeof(size_t) + b.size() * sizeof(int));

  std::map<int, double> c;
  c[8] = 1.1;
  c[9] = 2.2;
  c[10] = 3.3;
  // in a 32-bit build, the size is "free" because it packs with the first
  //  int key
  std::map<int, double> map_output;
  do_test("map<int,double>", c, map_output,
          (sizeof(size_t) + c.size() * (sizeof(int) + sizeof(double))));

  std::vector<std::string> ss;
  ss.push_back("Hello");
  ss.push_back("World");
  std::vector<std::string> vector_string_output;
  // sizeof(size_t): size of the vector
  // sizeof(unsigned): size of the 1st string
  // sizeof(char)*ss[0].size(): Hello, which is 5
  // sizeof(unsigned): size of the 2nd string
  // sizeof(char)*ss[1].size(): World, which is 5
  do_test("vector<string>", ss, vector_string_output,
          sizeof(size_t) + sizeof(unsigned) + sizeof(char) * ss[0].size() +
              sizeof(unsigned) + sizeof(char) * ss[1].size());

  std::set<int> s;
  s.insert(4);
  s.insert(2);
  s.insert(11);
  std::set<int> set_int_output;
  do_test("set<int>", s, set_int_output, sizeof(size_t) + s.size() * sizeof(int));

  if(error_count > 0) {
    std::cout << "ERRORS FOUND" << std::endl;
    exit(1);
  } else {
    std::cout << "all tests passed" << std::endl;
    exit(0);
  }
}
