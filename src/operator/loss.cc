/*!
 * Copyright (c) 2015 by Contributors
 * \file sum.cc
 * \brief
 * \author Luke Metz
*/

#include "./loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(LossParam param) {
  return new LossOp<cpu>(param);
}

Operator *LossProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(LossParam);

MXNET_REGISTER_OP_PROPERTY(Loss, LossProp)
.describe("Convert a tensor to a loss tensor. Returns a tensor with shape (1,)")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(LossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
