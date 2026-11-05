// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.
#pragma once
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace torchscan {
using J = nlohmann::json;
using N = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>, boost::multiprecision::et_off>;
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Uncounted : std::runtime_error {
  Uncounted() : std::runtime_error("No registered formula.") {}
};
inline constexpr auto METHOD = "ATen.RecordFunction/torchscan_flops_v1";
inline constexpr auto SCOPE = "calling_thread_and_autograd_eager_ops";

// Binary JSON values are internal exact integers, never objects accepted from the wire.
inline J number(const N& n) {
  if (n >= INT64_MIN && n <= INT64_MAX) return n.convert_to<int64_t>();
  if (n >= 0 && n <= UINT64_MAX) return n.convert_to<uint64_t>();
  auto text = n.str();
  return J::binary(std::vector<uint8_t>(text.begin(), text.end()));
}
inline N integer(const J& value) {
  if (value.is_number_unsigned()) return N(value.get<uint64_t>());
  if (value.is_number_integer()) return N(value.get<int64_t>());
  if (value.is_binary()) {
    const auto& bytes = value.get_binary();
    return N(std::string(bytes.begin(), bytes.end()));
  }
  throw Unsupported("Expected an integer.");
}
inline N natural(const J& value) {
  N n = integer(value);
  if (n < 0) throw Unsupported("Expected a nonnegative integer.");
  return n;
}
inline const J& array(const J& value) {
  if (!value.is_array()) throw Unsupported("Expected shape or argument array.");
  return value;
}
inline N product(const J& shape, size_t start = 0, size_t end = SIZE_MAX) {
  array(shape);
  if (start > shape.size()) throw Unsupported("Invalid shape rank.");
  N result = 1;
  for (size_t i = start; i < std::min(end, shape.size()); ++i) result *= natural(shape[i]);
  return result;
}
inline const J& get(const J& object, std::string_view key) {
  static const J none;
  auto found = object.is_object() ? object.find(key) : object.end();
  return found == object.end() ? none : *found;
}
inline const J& arg(const J& args, const J& kw, size_t index, std::string_view name) {
  return index < args.size() ? args.at(index) : get(kw, name);
}
inline double scalar(const J& value, double fallback = 0) {
  if (value.is_null()) return fallback;
  if (value.is_boolean()) return value.get<bool>();
  if (value.is_number()) return value.get<double>();
  if (value.is_binary()) return integer(value).convert_to<double>();
  if (value.is_object() && value.size() == 1) {
    auto tag = get(value, "nonfinite");
    if (tag == "inf") return INFINITY;
    if (tag == "-inf") return -INFINITY;
    if (tag == "nan") return NAN;
  }
  throw Unsupported("Unsupported scalar option.");
}
inline bool boolean(const J& value, bool fallback = false) {
  if (value.is_null()) return fallback;
  if (!value.is_boolean()) throw Unsupported("Expected a boolean option.");
  return value.get<bool>();
}
inline size_t axis(const J& shape, const J& dim) {
  N d = dim.is_null() ? N(-1) : integer(dim);
  if (d < 0) d += shape.size();
  if (d < 0 || d >= shape.size()) throw Unsupported("Invalid dimension.");
  return d.convert_to<size_t>();
}
struct Rule {
  std::string kind;
  J params;
};
const std::map<std::string, Rule>& rules();
N formula(const std::string& name, const J& args, const J& kwargs, const J& output);
J analyze(const J& capture);
J parse_json(const std::string& text);
void write_json(std::ostream& output, const J& value);
}  // namespace torchscan
