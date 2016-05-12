#include "runtests.h"
#include "testtasks.h"

template <bool PREDVAL>
TestResult test_pred_const(Runtime *runtime, Context ctx, const Globals& g)
{
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 45);
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 46,
		(PREDVAL ? Predicate::TRUE_PRED : Predicate::FALSE_PRED));
  Future f = CheckEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 
			     (PREDVAL ? 46 : 45));
  bool b = f.get_result<bool>();
  return b ? RESULT_PASS : RESULT_FAIL;
}

namespace {
  // avoids name conflicts on these with other files
  TestInformation pcf("pred_const_false", test_pred_const<false>, 1, 1, 1);
  TestInformation pct("pred_const_true", test_pred_const<false>, 1, 1, 1);
};

template <bool PREDVAL, bool EARLY>
TestResult test_pred_simple(Runtime *runtime, Context ctx, const Globals& g)
{
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 45);
  DelayedPredicate dp(runtime, ctx);
  if(EARLY)
    dp = PREDVAL;
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 46, dp);
  Future f = CheckEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 
			     (PREDVAL ? 46 : 45));
  if(!EARLY)
    dp = PREDVAL;
  bool b = f.get_result<bool>();
  return b ? RESULT_PASS : RESULT_FAIL;
}

namespace {
  TestInformation psf("pred_simple_false", test_pred_simple<false, false>, 1, 1, 1);
  TestInformation pst("pred_simple_true", test_pred_simple<true, false>, 1, 1, 1);
  TestInformation psfe("pred_simple_false_early", test_pred_simple<false, true>, 1, 1, 1);
  TestInformation pste("pred_simple_true_early", test_pred_simple<true, true>, 1, 1, 1);
}
