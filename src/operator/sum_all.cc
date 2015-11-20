/*!
 * Copyright (c) 2015 by Contributors
 * \file sum.cc
 * \brief
 * \author Luke Metz
*/

#include "./sum_all-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SumAllParam param) {
  return new SumAllOp<cpu>(param);
}

Operator *SumAllProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SumAllParam);

MXNET_REGISTER_OP_PROPERTY(SumAll, SumAllProp)
.describe("Sum all elements in a tensor. Returns a tensor with shape (1,)")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(SumAllParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
