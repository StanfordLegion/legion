//
//  logger_message_descriptor.h
//  Legion
//
//  Created by Alan Heirich on 5/5/17.
//
//

#ifndef LoggerMessageDescriptor_h
#define LoggerMessageDescriptor_h

#include "realm/logging.h"
#include "runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ostream>

#define REPORT_LEGION_FATAL(code, fmt, ...)               \
{                                                             \
char message[4096];                                         \
snprintf(message, 4096, fmt, ##__VA_ARGS__);                \
Legion::Internal::Runtime::report_fatal_message(            \
code, __FILE__, __LINE__, message);               \
}

#define REPORT_LEGION_ERROR(code, fmt, ...)               \
{                                                             \
char message[4096];                                         \
snprintf(message, 4096, fmt, ##__VA_ARGS__);                \
Legion::Internal::Runtime::report_error_message(            \
code, __FILE__, __LINE__, message);               \
}

#define REPORT_LEGION_WARNING(code, fmt, ...)             \
{                                                             \
char message[4096];                                         \
snprintf(message, 4096, fmt, ##__VA_ARGS__);                \
Legion::Internal::Runtime::report_warning_message(          \
code, __FILE__, __LINE__, message);               \
}


#endif /* logger_message_descriptor.h */
