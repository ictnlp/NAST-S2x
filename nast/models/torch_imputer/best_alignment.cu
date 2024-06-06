// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Licensed under the BSD-3-Clause license
// This is the GPU implementation of the Connectionist Temporal Loss.
// We mostly follow Graves.
// 1. Graves et al: http://www.cs.toronto.edu/~graves/icml_2006.pdf
// We use the equations from above link, but note that [1] has 1-based indexing
// and we (of course) use 0-based. Graves et al call the probabilities y, we use
// log_probs (also calling them inputs) A few optimizations (simmilar to those
// here, but also some I didn't take) are described in
// 2. Minmin Sun:
// http://on-demand.gputechconf.com/gtc/2016/presentation/s6383-minmin-sun-speech-recognition.pdf

#include <ATen/TensorUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <numeric>
#include <type_traits>

using namespace at;

// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1])
// so if l is l_0 l_1 ... l_(tl-1) then this looks up idx in
// l' = BLANK l_0 BLANK l_1 BLANK ... BLANK l_(tl-1) BLANK
// - note that no bound-checking is done
// - it is important to only call it witth idx == 0 if the target length is 0
// - __restrict__ impact to be measured, see
//   https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
template <typename target_t>
__device__ static inline int64_t
get_target_prime(const target_t *__restrict__ target, int64_t offset,
                 int64_t stride, int64_t idx, int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

// this kernel is a relatively straightforward implementation of the alpha
// calculation in the forward backward algorithm (section 4.1). A (minor) twist
// is that we are using log-calculations to enhance numerical stability
// (log_probs and log_alpha). In total it would be more efficient to compute the
// beta in the same kernel (e.g. cudnn does this). While the beta are not needed
// for the loss itself (just the grad), we can return log_alpha+log_beta (so
// same space as currently) and the overhead is small and the use-case for loss
// without grad is relatively limited. We parallelize by batch and target
// sequence. Empirically, it is faster to loop over the input (log probs)
// sequence  and do target in parallel, even if it means more frequent
// __syncthreads. In contrast to the cuDNN implementation, we allow large target
// lengths. For this we need that all previous `s` have been computed when we
// start a new block_s. This is why we have our own for loop here.
template <typename scalar_t, typename target_t>
__global__ void
#if defined(__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_2((std::is_same<scalar_t, float>::value ? 1024 : 896), 1)
#endif
    ctc_alignment_log_alpha_gpu_kernel(
        scalar_t *__restrict__ log_alpha_data, int64_t *__restrict__ paths_data,
        const scalar_t *log_probs_data,
        const int64_t *__restrict__ input_lengths, int64_t max_input_length,
        const target_t *__restrict__ targets_data,
        const int64_t *__restrict__ target_lengths, int64_t max_target_length,
        scalar_t *__restrict__ neg_log_likelihood_data, int64_t lp_input_stride,
        int64_t lp_batch_stride, int64_t lp_char_stride,
        int64_t la_batch_stride, int64_t la_input_stride,
        int64_t la_target_stride, const int64_t *__restrict__ tg_batch_offsets,
        int64_t tg_target_stride, int64_t batch_size, int64_t BLANK) {

  constexpr scalar_t neginf = -INFINITY;

  // bookkeeping
  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t lp_batch_offset = b * lp_batch_stride;
  int64_t la_batch_offset = b * la_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (b >= batch_size)
    return;

  // first row (t=0), the three equations for alpha_1 above eq (6)
  for (int64_t block_s = 0; block_s < 2 * max_target_length + 1;
       block_s += blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    scalar_t la;
    switch (s) {
    case 0:
      la = log_probs_data[lp_batch_offset + lp_char_stride * BLANK];
      break;
    case 1:
      la = target_length == 0
               ? neginf
               : log_probs_data[lp_batch_offset +
                                lp_char_stride *
                                    get_target_prime(
                                        targets_data, tg_batch_offset,
                                        tg_target_stride, 1, BLANK)];
      break;
    default:
      la = neginf;
    }
    if (s < 2 * max_target_length + 1)
      log_alpha_data[la_batch_offset +
                     /* la_input_stride * 0 */ +la_target_stride * s] = la;
  }

  for (int64_t block_s = 0; block_s < 2 * max_target_length + 1;
       block_s += blockDim.x) {
    int64_t s = threadIdx.x + block_s;

    // These two only depend on s, so we can cache them.
    int64_t current_char; // l_s in eq (6)
    bool have_three;      // flag which of the two cases in eq (6) we have
    if (s < 2 * target_length + 1 && target_length > 0) {
      current_char = get_target_prime(targets_data, tg_batch_offset,
                                      tg_target_stride, s, BLANK);
      have_three = ((s > 1) && (get_target_prime(targets_data, tg_batch_offset,
                                                 tg_target_stride, s - 2,
                                                 BLANK) != current_char));
    } else {
      current_char = BLANK;
      have_three = false;
    }
    for (int64_t t = 1; t < max_input_length; t++) {
      __syncthreads(); // on cuda 9 we might use partial synchronization of only
                       // the threads within the same batch
      if ((t < input_length) && (s < 2 * target_length + 1)) {
        // only for valid t, s. This is equation (6) and (7), la1, la2, la3 are
        // the three summands, lamax is the maximum for the logsumexp trick.
        scalar_t la1 =
            log_alpha_data[la_batch_offset + la_input_stride * (t - 1) +
                           la_target_stride * s];
        scalar_t lamax = la1;
        int64_t max_path = s;
        scalar_t la2, la3;
        if (s > 0) {
          la2 = log_alpha_data[la_batch_offset + la_input_stride * (t - 1) +
                               la_target_stride * (s - 1)];
          if (la2 > lamax) {
            lamax = la2;
            max_path = s - 1;
          }
        } else {
          la2 = neginf;
        }
        if (have_three) {
          la3 = log_alpha_data[la_batch_offset + la_input_stride * (t - 1) +
                               la_target_stride * (s - 2)];
          if (la3 > lamax) {
            lamax = la3;
            max_path = s - 2;
          }
        } else {
          la3 = neginf;
        }
        /*if (lamax == neginf) // when all are neginf. (then the whole thing is
                             // neginf, but we can pretend)
          lamax = 0;*/

        int64_t log_alpha_i =
            la_batch_offset + la_input_stride * t + la_target_stride * s;
        int64_t log_prob_i = lp_batch_offset + t * lp_input_stride +
                             lp_char_stride * current_char;

        log_alpha_data[log_alpha_i] = lamax + log_probs_data[log_prob_i];

        paths_data[log_alpha_i] = max_path;
      } else {
        // otherwise we just set to neginf
        if (s < 2 * max_target_length + 1)
          log_alpha_data[la_batch_offset + la_input_stride * t +
                         la_target_stride * s] = neginf;
      }
    }
  }
  __syncthreads(); // on cuda 9 we might use partial synchronization of only the
                   // threads within the same batch

  // compute the loss (eq (8))
  if (threadIdx.x == 0) {
    scalar_t l1 =
        log_alpha_data[la_batch_offset + la_input_stride * (input_length - 1) +
                       la_target_stride * (target_length * 2)];
    scalar_t l2 =
        target_length > 0
            ? log_alpha_data[la_batch_offset +
                             la_input_stride * (input_length - 1) +
                             la_target_stride * (target_length * 2 - 1)]
            : neginf;
    scalar_t m = ((l1 > l2) ? l1 : l2);
    m = ((m == neginf) ? 0 : m);
    scalar_t log_likelihood = std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
    neg_log_likelihood_data[b] = -log_likelihood;
  }
}

// The forward computation. Lot's of admin and a call to the alpha kernel.
// Note: we do not check that the labels are in the valid range. As we use
// them for indexing in the kernels, you'll see memory errors when you
// pass corrupt labels.
// We support both a 2-dimensional tensor as targets (one set of targets in each
// row) and a 1-dimensional tensor where all targets are concatenated (and we
// use target_lengths to figure out where they begin). We return log_alpha
// (currently, might change to (log_alpha+log_beta) to be passed to the
// backward. The dispatch function will only return the loss.
template <typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor, Tensor>
best_alignment_gpu_template(const Tensor &log_probs, const Tensor &targets,
                            IntArrayRef input_lengths,
                            IntArrayRef target_lengths, int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)
  CheckedFrom c = "ctc_alignment_gpu";
  using target_t =
      typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  checkAllSameGPU(c, {log_probs_arg, targets_arg});

  checkScalarType(c, targets_arg, target_scalar_type);
  checkDim(c, log_probs_arg, 3);
  checkDimRange(c, targets_arg, 1, 3);

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels),
              "blank must be in label range");
  TORCH_CHECK(input_lengths.size() == batch_size,
              "input_lengths must be of size batch_size");
  TORCH_CHECK(target_lengths.size() == batch_size,
              "target_lengths must be of size batch_size");

  int64_t lp_input_stride = log_probs.stride(0);
  int64_t lp_char_stride = log_probs.stride(2);
  int64_t tg_target_stride;

  int64_t max_target_length = 0;
  auto tg_batch_offsets =
      at::empty({batch_size}, at::device(at::kCPU).dtype(at::kLong));
  auto tg_batch_offsets_data = tg_batch_offsets.data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
    checkSize(c, targets_arg, 0, pos);
  } else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = i * tg_batch_stride;
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(1);
    checkSize(c, targets_arg, 0, batch_size);
    TORCH_CHECK(targets.size(1) >= max_target_length,
                "Expected tensor to have size at least ", max_target_length,
                " at dimension 1, but got size ", targets.size(1), " for ",
                targets_arg, " (while checking arguments for ", c, ")");
  }
  int64_t max_input_length = log_probs.size(0);
  for (int64_t b = 0; b < batch_size; b++) {
    TORCH_CHECK(input_lengths[b] <= max_input_length,
                "Expected input_lengths to have value at most ",
                max_input_length, ", but got value ", input_lengths[b],
                " (while checking arguments for ", c, ")");
  }

  auto target_lengths_t =
      at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t =
      at::tensor(input_lengths, targets.options().dtype(kLong));
  tg_batch_offsets = tg_batch_offsets.cuda();

  Tensor log_alpha =
      at::empty({batch_size, log_probs.size(0), 2 * max_target_length + 1},
                log_probs.options());
  Tensor paths = at::full_like(log_alpha, -1, log_alpha.options().dtype(kLong));
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  // Very likely, we could be more clever here, e.g. learning (or genralizing
  // and reusing) from SoftMax.cu...
  constexpr int max_threads =
      std::is_same<scalar_t, float>::value
          ? 512
          : 896; // we need 72 or so 32 bit registers for double
  int threads_target = max_threads;
  while (threads_target / 2 >= 2 * max_target_length + 1) {
    threads_target /= 2;
  }
  int threads_batch = std::min(max_threads / threads_target, (int)batch_size);
  dim3 block(threads_target, threads_batch);
  dim3 grid((2 * max_target_length + 1 + threads_target - 1) / threads_target,
            (batch_size + threads_batch - 1) / threads_batch);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ctc_alignment_log_alpha_gpu_kernel<scalar_t, target_t>
      <<<grid, block, 0, stream>>>(
          log_alpha.data_ptr<scalar_t>(), paths.data_ptr<int64_t>(),
          log_probs.data_ptr<scalar_t>(), input_lengths_t.data_ptr<int64_t>(),
          log_probs.size(0), targets.data_ptr<target_t>(),
          target_lengths_t.data_ptr<int64_t>(), max_target_length,
          neg_log_likelihood.data_ptr<scalar_t>(), log_probs.stride(0),
          log_probs.stride(1), log_probs.stride(2), log_alpha.stride(0),
          log_alpha.stride(1), log_alpha.stride(2),
          tg_batch_offsets.data_ptr<int64_t>(), tg_target_stride, batch_size,
          BLANK);
  AT_CUDA_CHECK(cudaGetLastError()); // catch launch errors
  return std::make_tuple(neg_log_likelihood, log_alpha, paths);
}

std::tuple<Tensor, Tensor, Tensor>
best_alignment_op(const Tensor &log_probs, const Tensor &targets,
                  IntArrayRef input_lengths, IntArrayRef target_lengths,
                  int64_t BLANK, bool zero_infinity) {
  (void)zero_infinity; // only used for backward
  return AT_DISPATCH_FLOATING_TYPES(
      log_probs.scalar_type(), "ctc_alignment_cuda", [&] {
        if (targets.scalar_type() == kLong) {
          return best_alignment_gpu_template<scalar_t, kLong>(
              log_probs, targets, input_lengths, target_lengths, BLANK);
        } else {
          return best_alignment_gpu_template<scalar_t, kInt>(
              log_probs, targets, input_lengths, target_lengths, BLANK);
        }
      });
}
