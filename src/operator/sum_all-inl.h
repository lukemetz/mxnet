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

namespace sum_all {
enum SumAllOpInputs {kData, kGamma};
enum SumAllOpOutputs {kOut, kMask};
// enum LeakyReLUOpType {kLeakyReLU, kPReLU, kRReLU};
// enum LeakyReLUOpResource {kRandom};
}  // namespace  sum

struct SumAllParam : public dmlc::Parameter<SumAllParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(SumAllParam) {
  }
};

template<typename xpu>
class SumAllOp : public Operator {
 public:
  explicit SumAllOp(SumAllParam param) {
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
    int size = in_data[sum_all::kData].Size();
    Tensor<xpu, 2> flat = in_data[sum_all::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 1> out = out_data[sum_all::kOut].get<xpu, 1, real_t>(s);
    Shape<2> s2 = Shape2(1, flat.shape_.Size());
    Assign(out, req[sum_all::kOut],
      sumall_except_dim<0>(reshape(flat, s2))
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
    std::cout << "ARRAYS" << std::endl;
    std::cout << in_grad[sum_all::kData].shape_ << std::endl;
    std::cout << out_grad[sum_all::kData].shape_ << std::endl;
    Tensor<xpu, 2> m_in_grad = in_grad[sum_all::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 1> m_out_grad = out_grad[sum_all::kData].get<xpu, 1, real_t>(s);
    std::cout << "START" << std::endl;
    Assign(m_in_grad, req[sum_all::kData],
      reshape(
        broadcast<0>(m_out_grad, Shape2(1, m_in_grad.shape_[0] * m_in_grad.shape_[1])),
        m_in_grad.shape_
      )
    );
    std::cout << "done" << std::endl;
  }

 private:
  SumAllParam param_;
};  // class SumAllOp

template<typename xpu>
Operator* CreateOp(SumAllParam type);

#if DMLC_USE_CXX11
class SumAllProp : public OperatorProperty {
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
    TShape oshape = Shape1(1);
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SumAllProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SumAll";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
      return {out_grad[sum_all::kOut]};
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
  SumAllParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUM_INL_H_
