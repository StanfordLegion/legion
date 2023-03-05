#ifndef __NATIVE_PYTHON_H__
#define __NATIVE_PYTHON_H__

#include "legion.h"

#ifdef __cplusplus
extern "C" {
#endif

// Used by canonical python to make the caller a implicit top level task
void legion_canonical_python_begin_top_level_task(int argc, char **argv);

// Used by canonical python to end the implicit top level task
void legion_canonical_python_end_top_level_task(void);

#ifdef __cplusplus
}
#endif

#endif