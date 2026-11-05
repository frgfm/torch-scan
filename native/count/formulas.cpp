// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.
#include "count.hpp"

namespace torchscan {
static N dim(const J& shape, size_t index) { return natural(array(shape).at(index)); }
static N gcd(N a, N b) {
  while (b != 0) {
    N remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}
static N convolution(const J& input, const J& weight, const J& out, bool transposed) {
  return 2 * dim(input, 0) * product(weight) * product(transposed ? input : out, 2);
}
static void attention_shapes(const J& q, const J& k, const J& v) {
  array(q);
  array(k);
  array(v);
  if (q.size() != 4 || k.size() != 4 || v.size() != 4 || q[0] != k[0] || k[0] != v[0] || k[1] != v[1] || k[2] != v[2] ||
      q[3] != k[3] || natural(k[1]) == 0 || natural(q[1]) % natural(k[1]) != 0)
    throw Unsupported("Attention requires matching batches, divisible query/KV heads, Q/K widths, and K/V lengths.");
}
static N mha(const J& q, const J& k, const J& v, const J& embed, const J& head, bool masked, bool averaged) {
  if (array(q).size() != 3 || q != k || k != v || q[2] != embed)
    throw Unsupported("Fused MHA counts require dense batched self-attention with matching widths.");
  N batch = natural(q[0]), length = natural(q[1]), width = natural(embed), heads = natural(head);
  N rows = batch * heads * length, scores = rows * length;
  return 8 * batch * length * width * width + batch * length * width + 4 * batch * length * length * width +
         5 * scores - 2 * rows + scores * (int(masked) + int(averaged));
}
N formula(const std::string& name, const J& args, const J& kw, const J& out) {
  array(args);
  auto found = rules().find(name);
  if (found == rules().end()) throw Uncounted();
  const auto& kind = found->second.kind;
  const auto& params = found->second.params;
  auto a = [&](size_t i) -> const J& {
    static const J none;
    return i < args.size() ? args.at(i) : none;
  };
  auto opt = [&](size_t i, const char* key) -> const J& { return arg(args, kw, i, key); };
  auto present = [&](size_t i, const char* key) { return int(!opt(i, key).is_null()); };
  auto cost = [&] { return int(scalar(get(params, "cost"), 1)); };
  if (kind == "_elementwise") return cost() * product(out);
  if (kind == "_activation") return cost() * product(a(0));
  if (kind == "_gelu") {
    auto approximation = get(kw, "approximate");
    if (!approximation.is_null() && approximation != "none" && approximation != "tanh")
      throw Unsupported("GELU estimates support only 'none' and 'tanh' approximations.");
    return (approximation == "tanh" ? 14 : 5) * product(out);
  }
  if (kind == "_weight_norm") {
    const auto& input = array(a(0));
    N axis = opt(2, "dim").is_null() ? N(0) : integer(opt(2, "dim")), rows;
    if (axis == -1)
      rows = 1;
    else if (axis >= 0 && axis < input.size())
      rows = natural(input.at(axis.convert_to<size_t>()));
    else
      throw Unsupported("WeightNorm FLOPs require a valid dimension or -1 for the whole tensor.");
    N n = product(input);
    if (product(a(1)) != rows || (rows != 0 && n == 0))
      throw Unsupported("WeightNorm FLOPs require one scale per nonempty norm row.");
    return 3 * n + rows;
  }
  if (kind == "_pool") {
    bool maximum = get(params, "maximum") == true;
    const auto& output = maximum && get(params, "indices") != false ? out.at(0) : out;
    const auto& kernel = opt(1, "kernel_size");
    N volume = 1;
    if (kernel.is_number() || (kernel.is_array() && kernel.size() == 1)) {
      N width = natural(kernel.is_array() ? kernel[0] : kernel);
      for (int i = 0; i < params["rank"].get<int>(); ++i) volume *= width;
    } else
      volume = product(kernel);
    return product(output) * (volume - int(maximum));
  }
  if (kind == "_adaptive_pool") {
    bool maximum = get(params, "maximum") == true;
    const auto& output = array(maximum ? out.at(0) : out);
    const auto& input = array(a(0));
    size_t rank = array(opt(1, "output_size")).size();
    if (!rank || rank > input.size() || rank > output.size()) throw Unsupported("Invalid adaptive pooling rank.");
    N visits = product(input, 0, input.size() - rank);
    for (size_t i = 0; i < rank; ++i) {
      N length = natural(input[input.size() - rank + i]), size = natural(output[output.size() - rank + i]);
      visits *= length + size - gcd(length, size);
    }
    return visits - product(output) * int(maximum);
  }
  if (kind == "_rms_norm") {
    const auto& input = array(a(0));
    const auto& normalized = array(opt(1, "normalized_shape"));
    bool valid = !normalized.empty() && normalized.size() <= input.size();
    if (valid)
      for (size_t i = 0; i < normalized.size(); ++i)
        valid &= natural(normalized[i]) > 0 && input[input.size() - normalized.size() + i] == normalized[i];
    if (!valid) throw Unsupported("RMSNorm estimates require matching nonempty normalized rows.");
    if (present(2, "weight") && opt(2, "weight") != normalized)
      throw Unsupported("RMSNorm FLOPs require a matching weight shape.");
    N n = product(input), rows = n / product(normalized);
    return (3 + present(2, "weight")) * n + 2 * rows;
  }
  if (kind == "_add") return product(out) * (1 + int(scalar(opt(2, "alpha"), 1) != 1));
  if (kind == "_add_product") return product(out) * (2 + int(scalar(opt(3, "value"), 1) != 1));
  if (kind == "_clamp") return product(out) * (present(1, "min") + present(2, "max"));
  if (kind == "_elu")
    return product(a(0)) * (6 + int(scalar(opt(2, "scale"), 1) != 1) + int(scalar(opt(3, "input_scale"), 1) != 1));
  if (kind == "_rrelu") {
    if (boolean(opt(4, "training"))) throw Unsupported("RReLU FLOPs cover evaluation calls only.");
    return 4 * product(a(0));
  }
  if (kind == "_softmax" || kind == "_log_softmax" || kind == "_safe_softmax" || kind == "_cumsum") {
    const auto& shape = array(a(0));
    N n = product(shape);
    N width = shape.empty() ? N(1) : natural(shape[axis(shape, opt(1, "dim"))]);
    if (n == 0 || width == 0) return 0;
    N rows = n / width;
    if (kind == "_cumsum") return n - rows;
    if (kind == "_log_softmax") return 4 * n - rows;
    return 5 * n - 2 * rows + (kind == "_safe_softmax" ? 2 * product(out) : N(0));
  }
  if (kind == "_sum" || kind == "_mean") {
    N n = product(a(0)), outputs = product(out);
    if (kind == "_mean" && n == 0 && outputs != 0) throw Unsupported("Mean FLOPs require nonempty reduction rows.");
    return std::max(N(0), n - outputs) + (kind == "_mean" ? outputs : N(0));
  }
  if (kind == "_layer_norm") {
    N n = product(a(0)), width = product(opt(1, "normalized_shape"));
    if (width == 0) throw Unsupported("LayerNorm FLOPs require a nonempty normalized row.");
    return 6 * n + 2 * (n / width) + n * (present(2, "weight") + present(3, "bias"));
  }
  if (kind == "_group_norm") {
    if (product(a(0), 1) == 0) throw Unsupported("GroupNorm FLOPs require a nonempty normalized group.");
    N n = product(a(0));
    return 6 * n + 2 * natural(opt(3, "N")) * natural(opt(6, "group")) +
           n * (present(1, "weight") + present(2, "bias"));
  }
  if (kind == "_batch_norm" || kind == "_batch_norm_legit" || kind == "_batch_norm_eval") {
    N n = product(a(0)), channels = dim(a(0), 1);
    if (n == 0) throw Unsupported("Empty native BatchNorm work depends on strides unavailable to the shape formula.");
    bool no_stats = kind == "_batch_norm_legit" && args.size() == 6;
    bool training = kind != "_batch_norm_eval" && boolean(opt(no_stats ? 3 : 5, "training"));
    N count = (2 + present(1, "weight") + present(2, "bias")) * n + 2 * channels;
    if (training)
      count += 4 * n + channels * (no_stats ? 0 : 3 * present(3, "running_mean") + 5 * present(4, "running_var"));
    return count;
  }
  if (kind == "_cpu_attention" || kind == "_attention") {
    bool backward = get(params, "backward") == true;
    const auto& q = a(int(backward));
    const auto& k = a(1 + int(backward));
    const auto& v = a(2 + int(backward));
    attention_shapes(q, k, v);
    N rows = product(q, 0, 3), scores = rows * natural(k[2]), qwidth = natural(q[3]), vwidth = natural(v[3]);
    if (kind == "_attention") return 2 * scores * (backward ? 3 * qwidth + 2 * vwidth : qwidth + vwidth);
    if (scalar(opt(3, "dropout_p")) != 0) throw Unsupported("CPU attention formula does not cover dropout.");
    if (scores == 0) return 0;
    return 2 * scores * (qwidth + vwidth) + 6 * scores - 2 * rows +
           scores * (int(boolean(opt(4, "is_causal"))) + int(!get(kw, "attn_mask").is_null()));
  }
  if (kind == "_native_mha")
    return mha(a(0), a(1), a(2), opt(3, "embed_dim"), opt(4, "num_head"), present(9, "mask"),
               boolean(opt(10, "need_weights"), true) && boolean(opt(11, "average_attn_weights"), true));
  if (kind == "_native_encoder") {
    N attention = mha(a(0), a(0), a(0), a(1), a(2), present(18, "mask"), false);
    N rows = product(a(0), 0, a(0).size() - 1), elements = product(a(0)), hidden = rows * dim(a(14), 0);
    return attention + 4 * hidden * natural(a(1)) + hidden * (boolean(a(7)) ? 5 : 1) + 18 * elements + 4 * rows;
  }
  if (kind == "_extremum") {
    if (a(1).is_array()) return product(out);
    const auto& output = out.is_array() && !out.empty() && out[0].is_array() ? out[0] : out;
    return std::max(N(0), product(a(0)) - product(output));
  }
  if (kind == "_norm") {
    const auto& order = kw.contains("ord") ? kw["ord"] : (kw.contains("p") ? kw["p"] : opt(1, "order"));
    double p = scalar(order, 2);
    N n = product(a(0));
    if (p == 2) return 2 * n;
    if (p == 1 || std::isinf(p)) return std::max(N(0), 2 * n - product(out));
    throw Unsupported("Norm FLOPs cover only orders 1, 2, and +/-infinity.");
  }
  if (kind == "_dot") return 2 * product(a(0));
  if (kind == "_grouped_mm") {
    const auto& input = array(a(0));
    const auto& weight = array(a(1));
    if (input.size() != 2 || weight.size() != 3 || input[1] != weight[1])
      throw Unsupported("Grouped matmul counts cover packed rows and equal-width expert matrices only.");
    return 2 * product(input) * natural(weight[2]);
  }
  if (kind == "_product") {
    const auto& left = array(a(params["left"].get<size_t>()));
    return 2 * product(out) * natural(left.at(left.size() - 1));
  }
  if (kind == "_convolution")
    return convolution(a(0), a(1), out,
                       params.contains("transposed_arg") && boolean(a(params["transposed_arg"].get<size_t>())));
  if (kind == "_convolution_backward")
    return convolution(a(1), a(2), a(0), boolean(a(7))) *
           (int(boolean(array(a(10)).at(0))) + int(boolean(a(10).at(1))));
  throw Uncounted();
}
}  // namespace torchscan
