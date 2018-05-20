#pragma once

#include "ATen/ATen.h"
#include <tuple>
#include <vector>

namespace at {
namespace native {

Tensor _cast_uint8_t(const Tensor & self, bool non_blocking=false);
Tensor _cast_int8_t(const Tensor & self, bool non_blocking=false);
Tensor _cast_double(const Tensor & self, bool non_blocking=false);
Tensor _cast_float(const Tensor & self, bool non_blocking=false);
Tensor _cast_int(const Tensor & self, bool non_blocking=false);
Tensor _cast_int64_t(const Tensor & self, bool non_blocking=false);
Tensor _cast_int16_t(const Tensor & self, bool non_blocking=false);
Tensor _cast_Half(const Tensor & self, bool non_blocking=false);
Tensor _cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional);
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state);
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask);
Tensor _cudnn_init_dropout_state(const Type & ty, double dropout, bool train, int64_t dropout_seed);
Tensor abs(const Tensor & self);
Tensor & abs_(Tensor & self);
Tensor & _abs_out_cpu(Tensor & result, const Tensor & self);
Tensor & _abs_out_cuda(Tensor & result, const Tensor & self);
Tensor adaptive_avg_pool1d(const Tensor & self, IntList output_size);
std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntList output_size);
bool allclose(const Tensor & self, const Tensor & other, double rtol=1e-05, double atol=1e-08, bool equal_nan=false);
Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
Tensor & addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
Tensor & addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
Tensor arange(const Type & dtype, Scalar start, Scalar end, Scalar step=1);
Tensor & arange_out(Tensor & result, Scalar start, Scalar end, Scalar step=1);
Tensor arange(const Type & dtype, Scalar end);
Tensor & arange_out(Tensor & result, Scalar end);
Tensor argmax(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor argmax(const Tensor & self);
Tensor _argmax(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor argmin(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor argmin(const Tensor & self);
Tensor _argmin(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled);
Tensor & bernoulli_(Tensor & self, const Tensor & p, Generator * generator=nullptr);
Tensor & bernoulli_(Tensor & self, double p=0.5, Generator * generator=nullptr);
Tensor bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias);
Tensor cat(TensorList tensors, int64_t dim=0);
Tensor & cat_out(Tensor & result, TensorList tensors, int64_t dim=0);
Tensor ceil(const Tensor & self);
Tensor & ceil_(Tensor & self);
Tensor & _ceil_out_cpu(Tensor & result, const Tensor & self);
Tensor & _ceil_out_cuda(Tensor & result, const Tensor & self);
std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim=0);
bool cudnn_is_acceptable(const Tensor & self);
Tensor convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups);
Tensor _convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled);
Tensor _convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding);
std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask);
Tensor conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1, int64_t groups=1);
Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1, int64_t groups=1);
Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1, int64_t groups=1);
Tensor conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad);
std::tuple<Tensor,Tensor,Tensor> conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad);
Tensor conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, int64_t groups=1, IntList dilation=1);
Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, int64_t groups=1, IntList dilation=1);
Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, int64_t groups=1, IntList dilation=1);
Tensor cos(const Tensor & self);
Tensor & cos_(Tensor & self);
Tensor & _cos_out_cpu(Tensor & result, const Tensor & self);
Tensor & _cos_out_cuda(Tensor & result, const Tensor & self);
Tensor cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin=0.0, bool size_average=true, bool reduce=true);
Tensor cudnn_affine_grid_generator_forward(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W);
Tensor cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W);
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon);
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon);
Tensor cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
Tensor cudnn_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
std::tuple<Tensor,Tensor,Tensor> cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask);
Tensor cudnn_convolution_backward_bias(const Tensor & grad_output);
Tensor cudnn_convolution_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
Tensor cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
std::tuple<Tensor,Tensor,Tensor> cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask);
Tensor cudnn_convolution_backward_bias(const Tensor & grad_output);
Tensor cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
Tensor cudnn_convolution_transpose_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
Tensor cudnn_grid_sampler_forward(const Tensor & self, const Tensor & grid);
std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output);
Tensor cumsum(const Tensor & self, int64_t dim, ScalarType dtype);
Tensor cumsum(const Tensor & self, int64_t dim);
Tensor & cumsum_out(Tensor & result, const Tensor & self, int64_t dim, ScalarType dtype);
Tensor & cumsum_out(Tensor & result, const Tensor & self, int64_t dim);
Tensor cumprod(const Tensor & self, int64_t dim, ScalarType dtype);
Tensor cumprod(const Tensor & self, int64_t dim);
Tensor & cumprod_out(Tensor & result, const Tensor & self, int64_t dim, ScalarType dtype);
Tensor & cumprod_out(Tensor & result, const Tensor & self, int64_t dim);
Tensor det(const Tensor & self);
Tensor diagflat(const Tensor & self, int64_t offset=0);
Tensor diagonal(const Tensor & self, int64_t offset=0);
Tensor dot(const Tensor & self, const Tensor & tensor);
Tensor einsum(std::string equation, TensorList tensors);
Tensor embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx=-1, bool scale_grad_by_freq=false, bool sparse=false);
Tensor embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
Tensor embedding_backward_cpu(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
Tensor embedding_backward_cuda(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
Tensor & embedding_renorm_cpu_(Tensor & self, const Tensor & indices, double max_norm, double norm_type);
Tensor & embedding_renorm_cuda_(Tensor & self, const Tensor & indices, double max_norm, double norm_type);
Tensor embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
std::tuple<Tensor,Tensor,Tensor> embedding_bag_cpu(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq=false, int64_t mode=0, bool sparse=false);
std::tuple<Tensor,Tensor,Tensor> embedding_bag_cuda(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq=false, int64_t mode=0, bool sparse=false);
Tensor embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse);
Tensor embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode);
Tensor embedding_bag_backward_cpu(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode);
Tensor embedding_bag_backward_cuda(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode);
Tensor empty(const Type & dtype, IntList size);
Tensor & empty_out(Tensor & result, IntList size);
Tensor empty_like(const Tensor & self);
Tensor empty_like(const Tensor & self, const Type & dtype);
Tensor exp(const Tensor & self);
Tensor & exp_(Tensor & self);
Tensor & _exp_out_cpu(Tensor & result, const Tensor & self);
Tensor & _exp_out_cuda(Tensor & result, const Tensor & self);
Tensor expand(const Tensor & self, IntList size);
Tensor expand_as(const Tensor & self, const Tensor & other);
Tensor eye(const Type & dtype, int64_t n, int64_t m=-1);
Tensor & eye_out_cpu(Tensor & result, int64_t n, int64_t m=-1);
Tensor & eye_out_cuda(Tensor & result, int64_t n, int64_t m=-1);
Tensor floor(const Tensor & self);
Tensor & floor_(Tensor & self);
Tensor & _floor_out_cpu(Tensor & result, const Tensor & self);
Tensor & _floor_out_cuda(Tensor & result, const Tensor & self);
Tensor full(const Type & dtype, IntList size, Scalar fill_value);
Tensor & full_out(Tensor & result, IntList size, Scalar fill_value);
Tensor full_like(const Tensor & self, Scalar fill_value);
Tensor full_like(const Tensor & self, Scalar fill_value, const Type & dtype);
Tensor hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin=1.0, bool size_average=true, bool reduce=true);
Tensor ger(const Tensor & self, const Tensor & vec2);
Tensor & ger_out(Tensor & result, const Tensor & self, const Tensor & vec2);
Tensor group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight={}, const Tensor & bias={}, double eps=1e-05, bool cudnn_enabled=true);
Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized=false);
Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized=false);
Tensor rfft(const Tensor & self, int64_t signal_ndim, bool normalized=false, bool onesided=true);
Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized=false, bool onesided=true, IntList signal_sizes={});
Tensor _fft_mkl(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes);
Tensor _fft_cufft(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes);
Tensor index(const Tensor & self, TensorList indices);
Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source);
Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values);
Tensor isclose(const Tensor & self, const Tensor & other, double rtol=1e-05, double atol=1e-08, bool equal_nan=false);
bool is_cuda(const Tensor & self);
bool is_distributed(const Tensor & self);
bool is_floating_point(const Tensor & self);
bool is_nonzero(const Tensor & self);
bool is_same_size(const Tensor & self, const Tensor & other);
bool is_signed(const Tensor & self);
bool is_sparse(const Tensor & self);
Tensor layer_norm(const Tensor & input, IntList normalized_shape, const Tensor & weight={}, const Tensor & bias={}, double eps=1e-05, bool cudnn_enable=true);
Tensor linspace(const Type & dtype, Scalar start, Scalar end, int64_t steps=100);
Tensor & linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps=100);
Tensor log(const Tensor & self);
Tensor & log_(Tensor & self);
Tensor & _log_out_cpu(Tensor & result, const Tensor & self);
Tensor & _log_out_cuda(Tensor & result, const Tensor & self);
Tensor logdet(const Tensor & self);
Tensor logspace(const Type & dtype, Scalar start, Scalar end, int64_t steps=100);
Tensor & logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps=100);
Tensor margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin=0.0, bool size_average=true, bool reduce=true);
Tensor matmul(const Tensor & self, const Tensor & other);
Tensor max_values(const Tensor & self, int64_t dim, bool keepdim=false);
std::tuple<Tensor,Tensor> max_pool1d(const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
Tensor min_values(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor mm(const Tensor & self, const Tensor & mat2);
Tensor & mm_out(Tensor & result, const Tensor & self, const Tensor & mat2);
Tensor mv(const Tensor & self, const Tensor & vec);
Tensor & mv_out(Tensor & result, const Tensor & self, const Tensor & vec);
Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length);
Tensor ones(const Type & dtype, IntList size);
Tensor & ones_out(Tensor & result, IntList size);
Tensor ones_like(const Tensor & self);
Tensor ones_like(const Tensor & self, const Type & dtype);
Tensor pairwise_distance(const Tensor & x1, const Tensor & x2, double p=2, double eps=1e-06, bool keepdim=false);
Tensor permute(const Tensor & self, IntList dims);
Tensor pin_memory(const Tensor & self);
Tensor rand(const Type & dtype, IntList size, Generator * generator=nullptr);
Tensor & rand_out(Tensor & result, IntList size, Generator * generator=nullptr);
Tensor rand_like(const Tensor & self);
Tensor rand_like(const Tensor & self, const Type & dtype);
Tensor randint(const Type & dtype, int64_t high, IntList size, Generator * generator=nullptr);
Tensor randint(const Type & dtype, int64_t low, int64_t high, IntList size, Generator * generator=nullptr);
Tensor & randint_out(Tensor & result, int64_t high, IntList size, Generator * generator=nullptr);
Tensor & randint_out(Tensor & result, int64_t low, int64_t high, IntList size, Generator * generator=nullptr);
Tensor randint_like(const Tensor & self, int64_t high);
Tensor randint_like(const Tensor & self, int64_t low, int64_t high);
Tensor randint_like(const Tensor & self, int64_t high, const Type & dtype);
Tensor randint_like(const Tensor & self, int64_t low, int64_t high, const Type & dtype);
Tensor randn(const Type & dtype, IntList size, Generator * generator=nullptr);
Tensor & randn_out(Tensor & result, IntList size, Generator * generator=nullptr);
Tensor randn_like(const Tensor & self);
Tensor randn_like(const Tensor & self, const Type & dtype);
Tensor randperm(const Type & dtype, int64_t n, Generator * generator=nullptr);
Tensor & randperm_out(Tensor & result, int64_t n, Generator * generator=nullptr);
Tensor range(const Type & dtype, Scalar start, Scalar end, Scalar step=1);
Tensor & range_out(Tensor & result, Scalar start, Scalar end, Scalar step=1);
Tensor repeat(const Tensor & self, IntList repeats);
Tensor reshape(const Tensor & self, IntList shape);
std::tuple<Tensor,Tensor> RoiPooling2d_forward_cpu(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale);
std::tuple<Tensor,Tensor> RoiPooling2d_forward_cuda(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale);
Tensor RoiPooling2d_backward_cpu(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes);
Tensor RoiPooling2d_backward_cuda(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes);
Tensor round(const Tensor & self);
Tensor & round_(Tensor & self);
Tensor & _round_out_cpu(Tensor & result, const Tensor & self);
Tensor & _round_out_cuda(Tensor & result, const Tensor & self);
Tensor rrelu(const Tensor & self, Scalar lower=0.125, Scalar upper=0.333333333333, bool training=false, Generator * generator=nullptr);
Tensor & rrelu_(Tensor & self, Scalar lower=0.125, Scalar upper=0.333333333333, bool training=false, Generator * generator=nullptr);
Tensor relu(const Tensor & self);
Tensor & relu_(Tensor & self);
Tensor select(const Tensor & self, int64_t dim, int64_t index);
Tensor selu(const Tensor & self);
Tensor & selu_(Tensor & self);
Tensor sin(const Tensor & self);
Tensor & sin_(Tensor & self);
Tensor & _sin_out_cpu(Tensor & result, const Tensor & self);
Tensor & _sin_out_cuda(Tensor & result, const Tensor & self);
int64_t size(const Tensor & self, int64_t dim);
Tensor slice(const Tensor & self, int64_t dim=0, int64_t start=0, int64_t end=9223372036854775807, int64_t step=1);
std::tuple<Tensor,Tensor> slogdet(const Tensor & self);
Tensor smm(const Tensor & self, const Tensor & mat2);
std::vector<Tensor> split(const Tensor & self, int64_t split_size, int64_t dim=0);
std::vector<Tensor> split_with_sizes(const Tensor & self, IntList split_sizes, int64_t dim=0);
Tensor squeeze(const Tensor & self);
Tensor squeeze(const Tensor & self, int64_t dim);
Tensor & squeeze_(Tensor & self);
Tensor & squeeze_(Tensor & self, int64_t dim);
Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
Tensor & _sspaddmm_out_only_sparse(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
Tensor & _sspaddmm_out_cpu(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
Tensor & _sspaddmm_out_cuda(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
Tensor stack(TensorList tensors, int64_t dim=0);
Tensor & stack_out(Tensor & result, TensorList tensors, int64_t dim=0);
Tensor stft(const Tensor & self, int64_t frame_length, int64_t hop, int64_t fft_size, bool normalized=false, bool onesided=true, const Tensor & window={}, int64_t pad_end=0);
int64_t stride(const Tensor & self, int64_t dim);
Tensor sum(const Tensor & self, ScalarType dtype);
Tensor sum(const Tensor & self);
Tensor _sum_cpu(const Tensor & self);
Tensor _sum_cuda(const Tensor & self);
Tensor sum(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype);
Tensor sum(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor sum(const Tensor & self, int64_t dim, ScalarType dtype);
Tensor _sum(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor & sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype);
Tensor & sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
Tensor & sum_out(Tensor & result, const Tensor & self, int64_t dim, ScalarType dtype);
Tensor & _sum_out_cpu(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
Tensor & _sum_out_cuda(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
Tensor sqrt(const Tensor & self);
Tensor & sqrt_(Tensor & self);
Tensor & _sqrt_out_cpu(Tensor & result, const Tensor & self);
Tensor & _sqrt_out_cuda(Tensor & result, const Tensor & self);
Tensor prod(const Tensor & self, ScalarType dtype);
Tensor prod(const Tensor & self);
Tensor _prod_cpu(const Tensor & self);
Tensor _prod_cuda(const Tensor & self);
Tensor prod(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype);
Tensor prod(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor prod(const Tensor & self, int64_t dim, ScalarType dtype);
Tensor _prod(const Tensor & self, int64_t dim, bool keepdim=false);
Tensor & prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype);
Tensor & prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
Tensor & prod_out(Tensor & result, const Tensor & self, int64_t dim, ScalarType dtype);
Tensor & _prod_out_cpu(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
Tensor & _prod_out_cuda(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
Tensor & t_(Tensor & self);
Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1);
Tensor _trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntList expand1, IntList expand2, IntList expand3, IntList sumdim, int64_t unroll_dim=1);
Tensor triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin=1.0, double p=2, double eps=1e-06, bool swap=false, bool size_average=true, bool reduce=true);
Tensor trunc(const Tensor & self);
Tensor & trunc_(Tensor & self);
Tensor & _trunc_out_cpu(Tensor & result, const Tensor & self);
Tensor & _trunc_out_cuda(Tensor & result, const Tensor & self);
Tensor type_as(const Tensor & self, const Tensor & other);
std::tuple<Tensor,Tensor> _unique_cpu(const Tensor & self, bool sorted=false, bool return_inverse=false);
std::tuple<Tensor,Tensor> _unique_cuda(const Tensor & self, bool sorted=false, bool return_inverse=false);
Tensor _unsafe_view(const Tensor & self, IntList size);
Tensor unsqueeze(const Tensor & self, int64_t dim);
Tensor & unsqueeze_(Tensor & self, int64_t dim);
Tensor view_as(const Tensor & self, const Tensor & other);
Tensor where(const Tensor & condition, const Tensor & self, const Tensor & other);
Tensor _s_where_cpu(const Tensor & condition, const Tensor & self, const Tensor & other);
Tensor _s_where_cuda(const Tensor & condition, const Tensor & self, const Tensor & other);
Tensor zeros(const Type & dtype, IntList size);
Tensor & zeros_out(Tensor & result, IntList size);
Tensor zeros_like(const Tensor & self);
Tensor zeros_like(const Tensor & self, const Type & dtype);
Tensor _standard_gamma_grad_cpu(const Tensor & self, const Tensor & output);
Tensor _standard_gamma_grad_cuda(const Tensor & self, const Tensor & output);
Tensor _s_poisson_cpu(const Tensor & self, Generator * generator=nullptr);
Tensor _s_poisson_cuda(const Tensor & self, Generator * generator=nullptr);
Tensor mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation);
Tensor mkldnn_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, bool bias_defined);
std::tuple<Tensor,Tensor> mkldnn_convolution_backward_weights(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, bool bias_defined);
std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, std::array<bool,3> output_mask);

}
}
