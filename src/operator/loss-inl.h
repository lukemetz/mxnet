/*!
 * Copyright (c) 2015 by Contributors
 * \file sum-inl.h
 * \brief sum
 * \author Luke Metz
*/
#ifndef MXNET_OPERATOR_SUM_INL_H_
#define MXNET_OPERATOR_SUM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace loss {
enum LossOpInputs {kData, kGamma};
enum LossOpOutputs {kOut, kMask};
// enum LeakyReLUOpType {kLeakyReLU, kPReLU, kRReLU};
// enum LeakyReLUOpResource {kRandom};
}  // namespace  sum

struct LossParam : public dmlc::Parameter<LossParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(LossParam) {
  }
};

template<typename xpu>
class LossOp : public Operator {
 public:
  explicit LossOp(LossParam param) {
    param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int size = in_data[loss::kData].Size();
    Tensor<xpu, 1> flat = in_data[loss::kData].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> out = out_data[loss::kOut].get<xpu, 1, real_t>(s);
    Assign(out, req[loss::kOut],
      F<mshadow_op::identity>(flat)
      );
  }

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1> m_in_grad = in_grad[loss::kData].get<xpu, 1, real_t>(s);
    Tensor<xpu, 0> m_out_grad = out_grad[loss::kData].get<xpu, 0, real_t>(s);
    Assign(m_in_grad, req[loss::kData], F<mshadow_op::identity>(m_in_grad) * 0 + 1);
  }

 private:
  LossParam param_;
};  // class LossOp

template<typename xpu>
Operator* CreateOp(LossParam type);

#if DMLC_USE_CXX11
class LossProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    out_shape->clear();
    //TShape oshape = Shape1(1);
    //out_shape->push_back(in_shape->at(loss::kData));
    out_shape->push_back(in_shape->at(loss::kData));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Loss";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
      return {};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
      return {};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return std::vector<ResourceRequest>();
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  LossParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUM_INL_H_
