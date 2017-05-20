// leaf tasks for CG solver

#ifndef CGTASKS_H
#define CGTASKS_H

#include "cgsolver.h"

class PrintField {
public:
  static void compute(const MyBlockMap& myblocks,
		      Runtime *runtime, Context ctx,
		      const char *prefix,
		      FieldID fid1, bool private1,
		      double minval = 0);

  static void preregister_tasks(void);

  //protected:
  static TaskID taskid;

  struct PrintFieldArgs {
    Rect<3> bounds;
    char prefix[32];
    double minval;
  };

  static void print_field_task(const Task *task,
			       const std::vector<PhysicalRegion> &regions,
			       Context ctx, Runtime *runtime);
};

class DotProduct {
public:
  static Future compute(const MyBlockMap& myblocks,
			DynamicCollective& dc_reduction,
			Runtime *runtime, Context ctx,
			FieldID fid1, bool private1,
			FieldID fid2, bool private2,
			Predicate pred = Predicate::TRUE_PRED);

  static void preregister_tasks(void);

  //protected:
  static TaskID taskid;

  struct DotpFieldArgs {
    Rect<3> bounds;
  };

  static double dotp_field_task(const Task *task,
				const std::vector<PhysicalRegion> &regions,
				Context ctx, Runtime *runtime);
};

// computes fid_sum = alpha1 * fid1 + alpha2 * fid2
class VectorAdd {
public:
  static void compute(const MyBlockMap& myblocks,
		      Runtime *runtime, Context ctx,
		      double alpha1, FieldID fid1, bool private1,
		      double alpha2, FieldID fid2, bool private2,
		      FieldID fid_sum, bool private_sum,
		      Predicate pred = Predicate::TRUE_PRED);

  static void preregister_tasks(void);

  //protected:
  static TaskID taskid;

  struct AddFieldArgs {
    Rect<3> bounds;
    double alpha1, alpha2;
  };

  static void add_field_task(const Task *task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx, Runtime *runtime);
};

// computes fid_acc = alpha1 * fid_acc + alpha2 * fid_in
class VectorAcc {
public:
  static void compute(MyBlockMap& myblocks,
		      Runtime *runtime, Context ctx,
		      double alpha_in, FieldID fid_in, bool private_in,
		      double alpha_acc, FieldID fid_acc, bool private_acc,
		      Predicate pred = Predicate::TRUE_PRED);

  static void preregister_tasks(void);

  //protected:
  static TaskID taskid;

  struct AccFieldArgs {
    Rect<3> bounds;
    double alpha_in, alpha_acc;
  };

  static void acc_field_task(const Task *task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx, Runtime *runtime);
};

#endif
