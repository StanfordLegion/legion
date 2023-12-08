#pragma once

#include "legion/region_tree.h"

namespace Legion {
namespace Internal {

class IndexSpaceValue
{
public:
  IndexSpaceValue() = delete;
  IndexSpaceValue(const IndexSpaceValue &) = delete;
  void operator=(const IndexSpaceValue &) = delete;

  // from an IndexSpaceExpression
  IndexSpaceValue(IndexSpaceExpression *expr_in);

  IndexSpaceValue(IndexSpaceValue &&other);
  IndexSpaceValue &operator=(IndexSpaceValue &&other);

  ~IndexSpaceValue();

  bool is_empty() const
  {
    if (expr != nullptr)
      return expr->is_empty();
    return dom.empty();
  }

  size_t get_volume() const
  {
    if (expr != nullptr)
      return expr->get_volume();
    return dom.get_volume();
  }

  IndexSpaceExpression *as_expr() const;
  IndexSpaceValue operator&(const IndexSpaceValue &other) const;
  IndexSpaceExpression *operator*() const
  {
    return as_expr();
  }

private:
  IndexSpaceValue(Domain dom, TypeTag type_tag);
  TypeTag type_tag;
  IndexSpaceExpression *expr;
  Domain dom;
};

} // namespace Internal
} // namepsace Legion
