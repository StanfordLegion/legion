#include "realm/faults.h"

#include <gtest/gtest.h>

using namespace Realm;

TEST(BacktraceTest, CapturePCs)
{
  Backtrace bt;

  bt.capture_backtrace();

  EXPECT_FALSE(bt.empty());
  EXPECT_GT(bt.hash(), 0);
  EXPECT_GT(bt.size(), 0);
}

TEST(BacktraceTest, Copy)
{
  Backtrace bt;
  bt.capture_backtrace();

  Backtrace bt_copy(bt);

  EXPECT_EQ(bt.size(), bt_copy.size());
  EXPECT_EQ(bt.hash(), bt_copy.hash());
  for(size_t i = 0; i < bt.size(); i++) {
    EXPECT_EQ(bt[i], bt_copy[i]);
  }
}

TEST(BacktraceTest, OperatorCopy)
{
  Backtrace bt;
  bt.capture_backtrace();

  Backtrace bt_copy = bt;

  EXPECT_EQ(bt.size(), bt_copy.size());
  EXPECT_EQ(bt.hash(), bt_copy.hash());
  for(size_t i = 0; i < bt.size(); i++) {
    EXPECT_EQ(bt[i], bt_copy[i]);
  }
}

TEST(BacktraceTest, OperatorEqual)
{
  Backtrace bt;
  bt.capture_backtrace();
  Backtrace bt_copy(bt);

  EXPECT_TRUE(bt.operator==(bt_copy));
}

TEST(BacktraceTest, Accessor)
{
  Backtrace bt;
  bt.capture_backtrace();
  Backtrace bt_copy(bt);
  std::vector<uintptr_t> pcs = bt.get_pcs();

  for(size_t i = 0; i < bt.size(); i++) {
    EXPECT_EQ(bt[i], pcs[i]);
  }
}

TEST(BacktraceTest, PrintSymbolString)
{
  Backtrace bt;
  bt.capture_backtrace();

  std::vector<std::string> symbols;
  bt.print_symbols(symbols);

  // it is possible that the resolved trace has more
  // frames than the raw one
  EXPECT_GT(symbols.size(), 0);
  EXPECT_GE(symbols.size(), bt.size());
}

TEST(BacktraceTest, PrintSymbolStream)
{
  Backtrace bt;
  bt.capture_backtrace();

  std::stringstream ss;
  bt.print_symbols(ss);

  // we verify stream version of print_symbols
  // with the string version
  std::vector<std::string> symbols_verification;
  bt.print_symbols(symbols_verification);
  std::vector<std::string> ss2str;
  std::string line;
  while(std::getline(ss, line)) {
    ss2str.push_back(line);
  }
  // the stream version has an extra print of "stack trace: N frames"
  EXPECT_EQ(ss2str.size() - 1, symbols_verification.size());
}

// we create a mock backtrace such that we can modify the pcs
class MockBacktrace : public Backtrace {
public:
  MockBacktrace(std::vector<uintptr_t> pcs_) { pcs = pcs_; }
};

TEST(BacktraceTest, Prune)
{
  Backtrace bt;
  bt.capture_backtrace();
  std::vector<uintptr_t> bt_pcs = bt.get_pcs();
  bt_pcs.erase(bt_pcs.begin()); // remove the first frame;
  MockBacktrace bt2(bt_pcs);

  bt.prune(bt2);

  EXPECT_EQ(bt.size(), 1);
}