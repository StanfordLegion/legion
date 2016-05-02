#include "runtests.h"
#include "testtasks.h"
#include "testmapper.h"

template <bool SPECVAL, bool ACTVAL>
TestResult test_spec_simple(Runtime *runtime, Context ctx, const Globals& g)
{
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 45);
  DelayedPredicate dp(runtime, ctx);
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 46, dp,
		(SPECVAL ? TestMapper::TAGOPT_SPECULATE_TRUE :
 		           TestMapper::TAGOPT_SPECULATE_FALSE));
  Future f = CheckEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 
			     (ACTVAL ? 46 : 45));
  dp = ACTVAL;
  bool b = f.get_result<bool>();
  return b ? RESULT_PASS : RESULT_FAIL;
}

namespace {
  TestInformation ssff("spec_simple_false_correct", test_spec_simple<false, false>, 1, 1, 1);
  TestInformation sstt("spec_simple_true_correct", test_spec_simple<true, true>, 1, 1, 1);
  TestInformation ssft("spec_simple_false_wrong", test_spec_simple<false, true>, 1, 1, 1);
  TestInformation sstf("spec_simple_true_wrong", test_spec_simple<true, false>, 1, 1, 1);
}
