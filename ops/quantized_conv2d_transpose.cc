
#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/padding.h"
// #include "tensorflow/core/kernels/quantization_utils.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// This interface is similar to the QuantizedConv2D op's interface:
// See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc#L2369
//
// Some discussions for the output_shape parameter:
// https://github.com/tensorflow/tensorflow/issues/2118
REGISTER_OP("QuantizedConv2DTranspose")
  .Input("value: Tinput")
  .Input("filter: Tfilter")
  .Input("output_sizes: int32")
  .Input("min_input: float")
  .Input("max_input: float")
  .Input("min_filter: float")
  .Input("max_filter: float")
  .Output("output: out_type")
  .Output("min_output: float")
  .Output("max_output: float")
  // .Attr("Tinput: quantizedtype")
  // .Attr("Tfilter: quantizedtype")
  // .Attr("out_type: quantizedtype = DT_QINT32") // default quantized type
  .Attr("Tinput: numbertype")
  .Attr("Tfilter: numbertype")
  .Attr("out_type: numbertype")
  .Attr("strides: list(int)")
  .Attr(GetPaddingAttrString()) // add padding string
  .SetShapeFn([] (InferenceContext *c) {
    // TODO: Don't understand this shape handle funtion for now, will check later
    // TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));

    ShapeHandle unused;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
    
    c->set_output(1, c->Scalar());
    c->set_output(2, c->Scalar());

    return Status::OK();
  })
  .Doc(R"doc(
Computes quantized conv2d_transpose.
)doc");


template <class T1, class T2, class T3>
class ReferenceConvTransFunctor {
  public:
    void operator() (OpKernelContext* context,
                     const T1* input_data,
                     int input_batches,
                     int input_height,
                     int input_width,
                     int input_depth,
                     int input_offset,
                     const T2* filter_data,
                     int filter_height,
                     int filter_width,
                     int filter_depth,
                     int filter_offset,
                     int stride,
                     Padding padding,
                     T3* output_data,
                     int output_height,
                     int output_width,
                     int output_shift,
                     int output_offset,
                     int output_mult)
  {
    // TODO: add support for padding
    // TODO: add support for stride
    
    int output_depth = filter_depth;

    for (int batch = 0; batch < input_batches; batch ++) {

      // for each channel in the output (which is the input of the conv2d)
      for (int c = 0; c < output_depth; c ++) {

        // we know that output_data is initialized as an array with zeros
        // h and w are the coordinate for an element in the gradient of output (input_data)
        for (int h = 0; h < input_height; h ++) {
          for (int w = 0; w < input_width; w ++) {
            // x and y are the coordinate of the center of the kernel that 
            // outputs the element at (h, w)
            int x = filter_height / 2 + h;
            int y = filter_width / 2 + w;

            for (int kx = 0; kx < filter_height; kx ++) {
              for (int ky = 0; ky < filter_width; ky ++) {
                int ox = x + kx - filter_height / 2;
                int oy = y + ky - filter_width / 2;

                T3 total = 0;
                for (int f = 0; f < input_depth; f++) {
                  const T1 input_value = input_data[
                    (batch * input_height * input_width * input_depth) +
                    (h * input_width * input_depth) +
                    (w * input_depth) +
                    (f)
                  ];

                  const T2 filter_value = filter_data[
                    (kx * filter_width * output_depth * input_depth) +
                    (ky * output_depth * input_depth) +
                    (c * input_depth) +
                    (f)
                  ];

                  // LOG(INFO) << "input_value  = " << input_value;
                  // LOG(INFO) << "filter_value  = " << filter_value;

                  total += input_value * filter_value;
                }
                
                output_data[
                  (batch * output_height * output_width * output_depth) +
                  (ox * output_width * output_depth) +
                  (oy * output_depth) +
                  (c)
                ] += total;

              }
            }


          }

        }
      }
    }

  }
};

template <class T1, class T2, class T3,
         template <class TF1, class TF2, class TF3> class ConvTransFunctor>
class QuantizedConv2DTransposeOp: public OpKernel {
  public:
    explicit QuantizedConv2DTransposeOp(OpKernelConstruction *context)
      : OpKernel(context) {
      //
      // Assertions for Attr

      // Assertions for strides
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      OP_REQUIRES(context, strides_.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must specify 4 dimensions"));
      OP_REQUIRES(context, strides_[1] == strides_[2],
                  errors::InvalidArgument("Current implementation only supports equal strides in the row and column dimensions"));
      OP_REQUIRES(context, (strides_[0] == 1) & (strides_[1] == 1),
                  errors::InvalidArgument("Current implementation doesn't support stride in batch or depth dimension"));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

      // These checkers are copied from the Conv2DFastBackpropInputOp
      // See: conv_grad_input_ops.cc
      // TODO: Assertions for data_format
      // Currently QuantizedConv2DTransposeOp doesn't support this parameter, could be implemented with
      // GetConvnetDataFormatAttrString()

      // The following checkers are set for the current implementation,
      // which are much stricter than the expected final release.
      // TODO: Remove these checkers when they are ready

      // Stride can only be 1 at the moment
      // OP_REQUIRES(context, strides_[1] == 1,
      //             errors::InvalidArgument("Current implementation only supports strides = {1,1,1,1}"));
    }

    void Compute(OpKernelContext* context) override {
      // fetch tensors from the context
      // input is the gradient w.r.t. to the output of the original convolution layer
      // shape [batch, in_rows, in_cols, in_depth]
      const Tensor& input = context->input(0); 

      // shape [filter_rows, filter_cols, out_depth, in_depth]
      const Tensor& filter = context->input(1);

      // 1-D Tensor
      const Tensor& output_sizes = context->input(2);

      LOG(INFO) << "Read tensors from context";

      // Check output_shape's dimension
      OP_REQUIRES(context,
                  TensorShapeUtils::IsVector(output_sizes.shape()),
                  errors::InvalidArgument("output_sizes should be a 1-D Tensor"));

      // Compute quantization related parameters
      const float min_input = context->input(3).flat<float>()(0);
      const float max_input = context->input(4).flat<float>()(0);
      const float min_filter = context->input(5).flat<float>()(0);
      const float max_filter = context->input(6).flat<float>()(0);

      LOG(INFO) << "min_input  = " << min_input;
      LOG(INFO) << "max_input  = " << max_input;
      LOG(INFO) << "min_filter = " << min_filter;
      LOG(INFO) << "max_filter = " << max_filter;

      // TODO: Use useful value here
      const int32 offset_input = 0;
          // FloatToQuantizedUnclamped<T1>(0.0f, min_input, max_input);
      const int32 offset_filter = 0;
          // FloatToQuantizedUnclamped<T2>(0.0f, min_filter, max_filter);
      const int32 offset_output = 0;
      const int32 mult_output = 1;
      const int32 shift_output = 0;

      // Get other constants
      const int64 in_depth = input.dim_size(3);
      OP_REQUIRES(context, in_depth == filter.dim_size(2),
          errors::InvalidArgument("input and filter doesn't match dimensions"));

      const int64 out_depth = filter.dim_size(3);

      const int64 input_rows = input.dim_size(1);
      const int64 filter_rows = filter.dim_size(0);
      const int64 out_rows = output_sizes.vec<int32>()(1);

      const int64 input_cols = input.dim_size(2);
      const int64 filter_cols = filter.dim_size(1);
      const int64 out_cols = output_sizes.vec<int32>()(2);

      const int64 batch = input.dim_size(0);

      const int stride = strides_[1];
      LOG(INFO) << "Created constants from Tensors and Attrs";

      // Create output_shape (TensorShape instance) from the input Tensor output_sizes
      TensorShape output_shape;
      OP_REQUIRES_OK(context,
                     TensorShapeUtils::MakeShape(output_sizes.vec<int32>(), &output_shape));

      Tensor *output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output)); 

      // Call the core computation
      ConvTransFunctor<T1, T2, T3> conv_trans_functor;
      conv_trans_functor(context,
                         input.flat<T1>().data(),
                         batch,
                         input_rows,
                         input_cols,
                         in_depth,
                         offset_input,
                         filter.flat<T2>().data(),
                         filter_rows,
                         filter_cols,
                         out_depth,
                         offset_filter,
                         stride,
                         padding_,
                         output->flat<T3>().data(),
                         out_rows,
                         out_cols,
                         shift_output,
                         offset_output,
                         mult_output);

      Tensor *min_output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, {}, &min_output));

      Tensor *max_output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(2, {}, &max_output));
    }

  private:
    std::vector<int32> strides_;
    Padding padding_;
};


REGISTER_KERNEL_BUILDER(
    Name("QuantizedConv2DTranspose")
      .Device(DEVICE_CPU)
      .TypeConstraint<float>("Tinput")
      .TypeConstraint<float>("Tfilter")
      .TypeConstraint<float>("out_type"),
    QuantizedConv2DTransposeOp<float, float, float, ReferenceConvTransFunctor>);
