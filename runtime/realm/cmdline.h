/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// command-line processing helpers

#ifndef REALM_CMDLINE_H
#define REALM_CMDLINE_H

#include "realm/realm_config.h"

#include <vector>
#include <string>

namespace Realm {

  class CommandLineOption;

  class REALM_PUBLIC_API CommandLineParser {
  public:
    CommandLineParser(void);
    ~CommandLineParser(void);

    template <typename T>
    CommandLineParser& add_option_int(const std::string& optname, T& target, bool keep = false);

    template <typename T>
    CommandLineParser& add_option_int_units(const std::string& optname, T& target,
					    char default_unit = 0, bool binary = true,
					    bool keep = false);

    template <typename T>
    CommandLineParser& add_option_string(const std::string& optname, T& target, bool keep = false);

    CommandLineParser& add_option_string(const std::string& optname, char *target, size_t maxlen, bool keep = false);

    template <typename T>
    CommandLineParser& add_option_stringlist(const std::string& optname, T& target, bool keep = false);

    CommandLineParser& add_option_bool(const std::string& optname, bool& target, bool keep = false);

    template <typename T>
    CommandLineParser& add_option_method(const std::string& optname, T *target,
					 bool (T::*method)(const std::string&), bool keep = false);

    bool parse_command_line(std::vector<std::string>& cmdline);
    bool parse_command_line(int argc, const char *argv[]);
    bool parse_command_line(int argc, char *argv[]);

  protected:
    std::vector<CommandLineOption *> options;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE CommandLineOption {
  public:
    CommandLineOption(const std::string& _optname, bool _keep);
    virtual ~CommandLineOption(void);

    virtual bool match(const std::string& s);
    virtual bool keep_arg(void) const;

    virtual bool parse_argument(std::vector<std::string>& cmdline,
				std::vector<std::string>::iterator& pos) = 0;
    virtual bool parse_argument(int& pos, int argc, const char *argv[]) = 0;

  protected:
    std::string optname;
    bool keep;
  };

  template <typename T>
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE IntegerCommandLineOption : public CommandLineOption {
  public:
    IntegerCommandLineOption(const std::string& _optname, bool _keep, T& _target);
    
    virtual bool parse_argument(std::vector<std::string>& cmdline,
				std::vector<std::string>::iterator& pos);
    virtual bool parse_argument(int& pos, int argc, const char *argv[]);

  protected:
    T& target;
  };

  template <typename T>
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE IntegerUnitsCommandLineOption : public CommandLineOption {
  public:
    IntegerUnitsCommandLineOption(const std::string& _optname,
				  char _default_unit, bool _binary,
				  bool _keep, T& _target);
    
    virtual bool parse_argument(std::vector<std::string>& cmdline,
				std::vector<std::string>::iterator& pos);
    virtual bool parse_argument(int& pos, int argc, const char *argv[]);

  protected:
    char default_unit;
    bool binary;
    T& target;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE StringCommandLineOption : public CommandLineOption {
  public:
    StringCommandLineOption(const std::string& _optname, bool _keep, std::string& _target);
    StringCommandLineOption(const std::string& _optname, bool _keep, char *_target, size_t _maxlen);
    
    virtual bool parse_argument(std::vector<std::string>& cmdline,
				std::vector<std::string>::iterator& pos);
    virtual bool parse_argument(int& pos, int argc, const char *argv[]);

  protected:
    std::string *target_str;
    char *target_array;
    size_t target_arrlen;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE StringListCommandLineOption : public CommandLineOption {
  public:
    StringListCommandLineOption(const std::string& _optname, bool _keep, std::vector<std::string>& _target);
    
    virtual bool parse_argument(std::vector<std::string>& cmdline,
				std::vector<std::string>::iterator& pos);
    virtual bool parse_argument(int& pos, int argc, const char *argv[]);

  protected:
    std::vector<std::string>& target;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE BooleanCommandLineOption : public CommandLineOption {
  public:
    BooleanCommandLineOption(const std::string& _optname, bool _keep, bool& _target);
    
    virtual bool parse_argument(std::vector<std::string>& cmdline,
				std::vector<std::string>::iterator& pos);
    virtual bool parse_argument(int& pos, int argc, const char *argv[]);

  protected:
    bool& target;
  };

  template <typename T>
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE MethodCommandLineOption : public CommandLineOption {
  public:
    MethodCommandLineOption(const std::string& _optname, bool _keep, T *_target,
			     bool (T::*_method)(const std::string&));
    
    virtual bool parse_argument(std::vector<std::string>& cmdline,
				std::vector<std::string>::iterator& pos);
    virtual bool parse_argument(int& pos, int argc, const char *argv[]);

  protected:
    T *target;
    bool (T::*method)(const std::string&);
  };

}; // namespace Realm

#include "realm/cmdline.inl"

#endif // ifndef REALM_CMDLINE_H

