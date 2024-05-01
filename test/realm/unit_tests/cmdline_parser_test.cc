#include "realm/cmdline.h"
#include "realm/realm_c.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class CommandLineCallback {
public:
  bool set_value(const std::string &str)
  {
    if(str == "yes") {
      value = "yes";
      return true;
    } else {
      return false;
    }
  }

  std::string value;
};

// we will test both old (version = 0) and new (version = 1) APIs

// success, remove the arguments that are parsed
TEST(CommandLineParserTest, SuccessNotKeep)
{
  for(int version = 0; version < 2; version++) {
    std::vector<std::string> cmdline = {"-testint",
                                        "1",
                                        "-testsizet_unit",
                                        "2",
                                        "-testbool",
                                        "-teststring",
                                        "string",
                                        "-teststring_list",
                                        "str1",
                                        "-teststring_list",
                                        "str2",
                                        "-testmethod",
                                        "yes"};
    int testint = 0;
    size_t testsizet_unit = 0;
    bool testbool = false;
    std::string teststring = "";
    std::vector<std::string> teststring_list;
    CommandLineCallback callback;
    CommandLineParser cp;
    cp.add_option_int("-testint", testint)
        .add_option_int_units("-testsizet_unit", testsizet_unit, 'M')
        .add_option_bool("-testbool", testbool)
        .add_option_string("-teststring", teststring)
        .add_option_stringlist("-teststring_list", teststring_list)
        .add_option_method<CommandLineCallback>("-testmethod", &callback,
                                                &CommandLineCallback::set_value);
    if(version == 0) {
      bool status = cp.parse_command_line(cmdline);
      EXPECT_EQ(status, true);
    } else {
      RealmStatus status = cp.parse_command_line_v2(cmdline);
      EXPECT_EQ(status, REALM_SUCCESS);
    }
    EXPECT_EQ(testint, 1);
    EXPECT_EQ(testsizet_unit, 2 * 1024 * 1024);
    EXPECT_EQ(testbool, true);
    EXPECT_EQ(teststring, "string");
    EXPECT_EQ(teststring_list.size(), 2);
    EXPECT_EQ(teststring_list[0], "str1");
    EXPECT_EQ(teststring_list[1], "str2");
    EXPECT_EQ(callback.value, "yes");
    EXPECT_EQ(cmdline.size(), 0);
  }
}

// success, keep the arguments that are parsed
TEST(CommandLineParserTest, SuccessKeep)
{
  for(int version = 0; version < 2; version++) {
    std::vector<std::string> cmdline = {"-testint",
                                        "1",
                                        "-testsizet_unit",
                                        "2",
                                        "-testbool",
                                        "-teststring",
                                        "string",
                                        "-teststring_list",
                                        "str1",
                                        "-teststring_list",
                                        "str2",
                                        "-testmethod",
                                        "yes"};
    int testint = 0;
    size_t testsizet_unit = 0;
    bool testbool = false;
    std::string teststring = "";
    std::vector<std::string> teststring_list;
    CommandLineCallback callback;
    CommandLineParser cp;
    cp.add_option_int("-testint", testint, true)
        .add_option_int_units("-testsizet_unit", testsizet_unit, 'M', true, true)
        .add_option_bool("-testbool", testbool, true)
        .add_option_string("-teststring", teststring, true)
        .add_option_stringlist("-teststring_list", teststring_list, true)
        .add_option_method<CommandLineCallback>("-testmethod", &callback,
                                                &CommandLineCallback::set_value, true);
    if(version == 0) {
      bool status = cp.parse_command_line(cmdline);
      EXPECT_EQ(status, true);
    } else {
      RealmStatus status = cp.parse_command_line_v2(cmdline);
      EXPECT_EQ(status, REALM_SUCCESS);
    }
    EXPECT_EQ(testint, 1);
    EXPECT_EQ(testsizet_unit, 2 * 1024 * 1024);
    EXPECT_EQ(testbool, true);
    EXPECT_EQ(teststring, "string");
    EXPECT_EQ(teststring_list.size(), 2);
    EXPECT_EQ(teststring_list[0], "str1");
    EXPECT_EQ(teststring_list[1], "str2");
    EXPECT_EQ(callback.value, "yes");
    EXPECT_EQ(cmdline.size(), 13);
  }
}

// success, but there are extra arguments that are not recognized
TEST(CommandLineParserTest, SuccessExtraArgs)
{
  for(int version = 0; version < 2; version++) {
    std::vector<std::string> cmdline = {"-testint", "1", "-unknown", "a"};
    int testint = 0;
    CommandLineParser cp;
    cp.add_option_int("-testint", testint);
    if(version == 0) {
      bool status = cp.parse_command_line(cmdline);
      EXPECT_EQ(status, true);
    } else {
      RealmStatus status = cp.parse_command_line_v2(cmdline);
      EXPECT_EQ(status, REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS);
    }
    EXPECT_EQ(testint, 1);
    EXPECT_EQ(cmdline.size(), 2);
  }
}

// in all the following failed tests, we will add a succeed argument before the failed one
// to make sure the succeed one can be passed correctly
// failed, did not pass a int
TEST(CommandLineParserTest, ErrorUnknownInterger)
{
  for(int version = 0; version < 2; version++) {
    std::vector<std::string> cmdline = {"-success", "123", "-testint", "abc"};
    int succeed_arg = 0;
    int testint = 0;
    CommandLineParser cp;
    cp.add_option_int("-success", succeed_arg).add_option_int("-testint", testint);
    if(version == 0) {
      bool status = cp.parse_command_line(cmdline);
      EXPECT_EQ(status, false);
    } else {
      RealmStatus status = cp.parse_command_line_v2(cmdline);
      EXPECT_EQ(status, REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER);
    }
    EXPECT_EQ(succeed_arg, 123);
    EXPECT_EQ(cmdline.size(), 2);
  }
}

// failed, did not pass a int
TEST(CommandLineParserTest, ErrorUnknownIntergerUnit)
{
  for(int version = 0; version < 2; version++) {
    std::vector<std::string> cmdline = {"-success", "123", "-testsizet_unit", "abc"};
    int succeed_arg = 0;
    size_t testsizet_unit = 0;
    CommandLineParser cp;
    cp.add_option_int("-success", succeed_arg)
        .add_option_int_units("-testsizet_unit", testsizet_unit, 'M');
    if(version == 0) {
      bool status = cp.parse_command_line(cmdline);
      EXPECT_EQ(status, false);
    } else {
      RealmStatus status = cp.parse_command_line_v2(cmdline);
      EXPECT_EQ(status, REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER_UNIT);
    }
    EXPECT_EQ(succeed_arg, 123);
    EXPECT_EQ(cmdline.size(), 2);
  }
}

// failed, did not pass a value
TEST(CommandLineParserTest, ErrorMissingInput)
{
  for(int version = 0; version < 2; version++) {
    {
      std::vector<std::string> cmdline = {"-success", "123", "-testint"};
      int succeed_arg = 0;
      int testint = 0;
      CommandLineParser cp;
      cp.add_option_int("-success", succeed_arg).add_option_int("-testint", testint);
      if(version == 0) {
        bool status = cp.parse_command_line(cmdline);
        EXPECT_EQ(status, false);
      } else {
        RealmStatus status = cp.parse_command_line_v2(cmdline);
        EXPECT_EQ(status, REALM_ARGUMENT_ERROR_MISSING_INPUT);
      }
      EXPECT_EQ(succeed_arg, 123);
      EXPECT_EQ(cmdline.size(), 1);
    }
    {
      std::vector<std::string> cmdline = {"-success", "123", "-testsizet_unit"};
      int succeed_arg = 0;
      size_t testsizet_unit = 0;
      CommandLineParser cp;
      cp.add_option_int("-success", succeed_arg)
          .add_option_int_units("-testsizet_unit", testsizet_unit, 'M');
      if(version == 0) {
        bool status = cp.parse_command_line(cmdline);
        EXPECT_EQ(status, false);
      } else {
        RealmStatus status = cp.parse_command_line_v2(cmdline);
        EXPECT_EQ(status, REALM_ARGUMENT_ERROR_MISSING_INPUT);
      }
      EXPECT_EQ(succeed_arg, 123);
      EXPECT_EQ(cmdline.size(), 1);
    }
    {
      std::vector<std::string> cmdline = {"-success", "123", "-teststring"};
      int succeed_arg = 0;
      std::string teststring = "";
      CommandLineParser cp;
      cp.add_option_int("-success", succeed_arg)
          .add_option_string("-teststring", teststring);
      if(version == 0) {
        bool status = cp.parse_command_line(cmdline);
        EXPECT_EQ(status, false);
      } else {
        RealmStatus status = cp.parse_command_line_v2(cmdline);
        EXPECT_EQ(status, REALM_ARGUMENT_ERROR_MISSING_INPUT);
      }
      EXPECT_EQ(succeed_arg, 123);
      EXPECT_EQ(cmdline.size(), 1);
    }
    {
      std::vector<std::string> cmdline = {"-success", "123", "-teststring_list"};
      int succeed_arg = 0;
      std::vector<std::string> teststring_list;
      CommandLineParser cp;
      cp.add_option_int("-success", succeed_arg)
          .add_option_stringlist("-teststring_list", teststring_list);
      if(version == 0) {
        bool status = cp.parse_command_line(cmdline);
        EXPECT_EQ(status, false);
      } else {
        RealmStatus status = cp.parse_command_line_v2(cmdline);
        EXPECT_EQ(status, REALM_ARGUMENT_ERROR_MISSING_INPUT);
      }
      EXPECT_EQ(succeed_arg, 123);
      EXPECT_EQ(cmdline.size(), 1);
    }
  }
}

// failed, the output char array is too short
TEST(CommandLineParserTest, ErrorOutputStringTooShort)
{
  for(int version = 0; version < 2; version++) {
    std::vector<std::string> cmdline = {"-success", "123", "-teststring", "toolong"};
    int succeed_arg = 0;
    size_t string_len = 2;
    char teststring[string_len];
    CommandLineParser cp;
    cp.add_option_int("-success", succeed_arg)
        .add_option_string("-teststring", teststring, string_len);
    if(version == 0) {
      bool status = cp.parse_command_line(cmdline);
      EXPECT_EQ(status, false);
    } else {
      RealmStatus status = cp.parse_command_line_v2(cmdline);
      EXPECT_EQ(status, REALM_ARGUMENT_ERROR_OUTPUT_STRING_TOO_SHORT);
    }
    EXPECT_EQ(succeed_arg, 123);
    EXPECT_EQ(cmdline.size(), 2);
  }
}

// failed, the callback method returns false
TEST(CommandLineParserTest, ErrorMethodReturnFalse)
{
  for(int version = 0; version < 2; version++) {
    std::vector<std::string> cmdline = {"-success", "123", "-testmethod", "no"};
    int succeed_arg = 0;
    CommandLineCallback callback;
    CommandLineParser cp;
    cp.add_option_int("-success", succeed_arg)
        .add_option_method<CommandLineCallback>("-testmethod", &callback,
                                                &CommandLineCallback::set_value);
    if(version == 0) {
      bool status = cp.parse_command_line(cmdline);
      EXPECT_EQ(status, false);
    } else {
      RealmStatus status = cp.parse_command_line_v2(cmdline);
      EXPECT_EQ(status, REALM_ARGUMENT_ERROR_METHOD_RETURN_FALSE);
    }
    EXPECT_EQ(succeed_arg, 123);
    EXPECT_EQ(cmdline.size(), 2);
  }
}
