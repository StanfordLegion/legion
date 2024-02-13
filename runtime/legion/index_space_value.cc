#include "legion/runtime.h"
#include "legion/index_space_value.h"

namespace Legion {
namespace Internal {

IndexSpaceValue::IndexSpaceValue(IndexSpaceExpression *expr_in)
  :type_tag(expr_in->type_tag), expr(nullptr)
{
  if (true) // TODO: restore domain path once perf issues are resolved
  {
    expr = expr_in;
    expr->add_base_expression_reference(INDEX_SPACE_VALUE_REF);
  }
  else
    expr_in->get_domain(dom);
}

IndexSpaceValue::IndexSpaceValue(Domain dom, TypeTag type_tag)
  :type_tag(type_tag), expr(nullptr), dom(dom)
{
}

IndexSpaceValue::IndexSpaceValue(IndexSpaceValue &&other)
  :type_tag(other.type_tag),
   expr(other.expr),
   dom(std::move(other.dom))
{
  other.expr = nullptr;
}

IndexSpaceValue &IndexSpaceValue::operator=(IndexSpaceValue &&other)
{
  type_tag = other.type_tag;

  expr = other.expr;
  other.expr = nullptr;

  dom = std::move(other.dom);

  return *this;
}

IndexSpaceValue::~IndexSpaceValue()
{
  if (expr != nullptr)
    if (expr->remove_base_expression_reference(INDEX_SPACE_VALUE_REF))
      delete expr;
  expr = nullptr;
}

IndexSpaceValue IndexSpaceValue::operator&(const IndexSpaceValue &other) const
{
  RegionTreeForest *forest = implicit_runtime->forest;

  if (expr != nullptr || other.expr != nullptr)
  {
    return IndexSpaceValue(forest->intersect_index_spaces(as_expr(),
                                                          other.as_expr()));
  }

  return IndexSpaceValue(dom.intersection(other.dom), type_tag);
}

IndexSpaceExpression *
IndexSpaceValue::as_expr() const
{
  if (expr != nullptr)
    return expr;

  return InternalExpressionCreator::create_with_domain(type_tag, dom);
}

} // namespace Internal
} // naemspace Legion
