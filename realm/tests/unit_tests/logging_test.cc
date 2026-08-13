/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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

#include "realm/logging.h"

#include <gtest/gtest.h>

#include <iomanip>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Realm {

  namespace {

    class CapturingLoggerOutput : public LoggerOutputStream {
    public:
      virtual void log_msg(Logger::LoggingLevel level, const char *name,
                           const char *msgdata, size_t msglen)
      {
        messages.push_back(std::string(msgdata, msglen));
      }

      virtual void flush(void) {}

      std::vector<std::string> messages;
    };

    class TestLogger : public Logger {
    public:
      TestLogger(const std::string &name, LoggerOutputStream *output)
        : Logger(name)
      {
        configured = true;
        log_level = LEVEL_SPEW;
        add_stream(output, LEVEL_SPEW, false /*delete_when_done*/,
                   false /*flush_each_write*/);
      }
    };

    class TestLoggerMessage : public LoggerMessage {
    public:
      TestLoggerMessage(Logger &logger, bool active)
        : LoggerMessage(&logger, active, Logger::LEVEL_INFO)
      {}

      TestLoggerMessage(TestLoggerMessage &&to_move)
        : LoggerMessage(std::move(to_move))
      {}
    };

    static_assert(!std::is_copy_constructible<LoggerMessage>::value,
                  "LoggerMessage must not be copy constructible");
    static_assert(!std::is_copy_assignable<LoggerMessage>::value,
                  "LoggerMessage must not be copy assignable");
    static_assert(std::is_move_constructible<LoggerMessage>::value,
                  "LoggerMessage must be move constructible");
    static_assert(!std::is_move_assignable<LoggerMessage>::value,
                  "LoggerMessage must not be move assignable");

    TEST(LoggerMessageTest, MoveTransfersBufferedMessage)
    {
      CapturingLoggerOutput output;
      TestLogger logger("move_buffered_message", &output);
      const std::string message(256, 'x');

      {
        TestLoggerMessage source(logger, true);
        source << message;

        TestLoggerMessage destination(std::move(source));

        EXPECT_FALSE(source.is_active());
        EXPECT_TRUE(destination.is_active());
      }

      ASSERT_EQ(output.messages.size(), 1U);
      EXPECT_EQ(output.messages.front(), message);
    }

    TEST(LoggerMessageTest, MovePreservesStreamFormatting)
    {
      CapturingLoggerOutput output;
      TestLogger logger("move_stream_formatting", &output);

      {
        TestLoggerMessage source(logger, true);
        source << "value=" << std::hex << std::showbase;

        TestLoggerMessage destination(std::move(source));
        destination << 42;
      }

      ASSERT_EQ(output.messages.size(), 1U);
      EXPECT_EQ(output.messages.front(), "value=0x2a");
    }

    TEST(LoggerMessageTest, MovePreservesInactiveState)
    {
      CapturingLoggerOutput output;
      TestLogger logger("move_inactive_message", &output);

      {
        TestLoggerMessage source(logger, true);
        source << "deactivated";
        source.deactivate();

        TestLoggerMessage destination(std::move(source));

        EXPECT_FALSE(source.is_active());
        EXPECT_FALSE(destination.is_active());
      }

      EXPECT_TRUE(output.messages.empty());
    }

  }; // anonymous namespace

}; // namespace Realm
