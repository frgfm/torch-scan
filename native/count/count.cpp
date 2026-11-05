// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.
#include "count.hpp"

#include "tables.hpp"

namespace torchscan {
const std::map<std::string, Rule>& rules() {
  static const auto result = [] {
    std::map<std::string, Rule> result;
    for (const auto& row : J::parse(RULES)) {
      std::istringstream names(row[2].get<std::string>());
      for (std::string name; names >> name;) result.emplace(name, Rule{row[0], row[1]});
    }
    return result;
  }();
  return result;
}
static const std::map<std::string, std::string>& ignored() {
  static const auto result = [] {
    std::map<std::string, std::string> result;
    for (const auto& row : J::parse(IGNORED)) {
      std::istringstream names(row[1].get<std::string>());
      for (std::string name; names >> name;) result.emplace(name, row[0]);
    }
    return result;
  }();
  return result;
}
// Keep integer tokens exact when nlohmann's standard 64-bit representation overflows.
class Parser : public nlohmann::json_sax<J> {
  J& root;
  std::vector<J*> stack;
  std::string object_key;
  J& put(J value) {
    if (stack.empty()) {
      root = std::move(value);
      return root;
    }
    if (stack.back()->is_array()) {
      stack.back()->push_back(std::move(value));
      return stack.back()->back();
    }
    auto& target = (*stack.back())[object_key];
    target = std::move(value);
    return target;
  }
  bool container(J value) {
    if (stack.size() >= 128) throw std::invalid_argument("Capture nesting exceeds 128 levels.");
    auto& target = put(std::move(value));
    stack.push_back(&target);
    return true;
  }

 public:
  explicit Parser(J& result) : root(result) {}
  bool null() override {
    put(nullptr);
    return true;
  }
  bool boolean(bool value) override {
    put(value);
    return true;
  }
  bool number_integer(int64_t value) override {
    put(value);
    return true;
  }
  bool number_unsigned(uint64_t value) override {
    put(value);
    return true;
  }
  bool number_float(double value, const std::string& token) override {
    if (token.find_first_of(".eE") == std::string::npos)
      put(J::binary(std::vector<uint8_t>(token.begin(), token.end())));
    else
      put(value);
    return true;
  }
  bool string(std::string& value) override {
    put(std::move(value));
    return true;
  }
  bool binary(J::binary_t& value) override {
    put(J::binary(std::move(value)));
    return true;
  }
  bool key(std::string& value) override {
    object_key = std::move(value);
    return true;
  }
  bool start_object(size_t) override { return container(J::object()); }
  bool start_array(size_t) override { return container(J::array()); }
  bool end_object() override {
    stack.pop_back();
    return true;
  }
  bool end_array() override {
    stack.pop_back();
    return true;
  }
  bool parse_error(size_t, const std::string&, const J::exception& error) override {
    throw std::invalid_argument(error.what());
  }
};
J parse_json(const std::string& text) {
  J result;
  Parser parser(result);
  if (!J::sax_parse(text, &parser)) throw std::invalid_argument("Invalid capture JSON.");
  return result;
}
void write_json(std::ostream& output, const J& value) {
  if (value.is_binary()) {
    output << integer(value);
    return;
  }
  if (value.is_array() || value.is_object()) {
    bool object = value.is_object(), first = true;
    output << (object ? '{' : '[');
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (!first) output << ',';
      first = false;
      if (object) output << J(it.key()).dump() << ':';
      write_json(output, *it);
    }
    output << (object ? '}' : ']');
  } else
    output << value.dump();
}
static J shapes(const J& value) {
  if (value.is_object()) {
    const auto& tag = get(value, "nonfinite");
    if (value.size() == 1 && (tag == "inf" || tag == "-inf" || tag == "nan")) return value;
    if (value.contains("unknown"))
      throw Unsupported(get(value, "unknown").is_string() ? value["unknown"].get<std::string>()
                                                          : "Unknown argument metadata.");
    const auto& dims = array(get(value, "shape"));
    if (!get(value, "floating").is_boolean() || !get(value, "complex").is_boolean() ||
        !get(value, "dtype").is_string() || !get(value, "device").is_string())
      throw Unsupported("Invalid tensor metadata.");
    for (const auto& dim : dims)
      if (natural(dim) > INT64_MAX) throw Unsupported("Invalid tensor dimension.");
    return dims;
  }
  if (value.is_array()) {
    J result = J::array();
    for (const auto& item : value) result.push_back(shapes(item));
    return result;
  }
  return value;
}
struct Flags {
  bool tensor = false, floating = false, complex = false;
  void add(const J& value) {
    if (value.is_object() && value.contains("shape")) {
      tensor = true;
      if (!get(value, "floating").is_boolean() || !get(value, "complex").is_boolean())
        throw Unsupported("Invalid tensor metadata.");
      floating |= value["floating"].get<bool>();
      complex |= value["complex"].get<bool>();
    } else if (value.is_object() || value.is_array())
      for (const auto& item : value) add(item);
  }
};
J analyze(const J& capture) {
  const auto& context = get(capture, "context");
  if (get(capture, "schema_version") != 1 || !get(capture, "schema_version").is_number_integer() ||
      get(capture, "report_type") != "operator_capture" || get(context, "method") != METHOD ||
      get(context, "scope") != SCOPE || get(context, "boundary") != "dispatcher_operator" ||
      !get(context, "torch_version").is_string() || !get(capture, "inputs").is_object() ||
      !get(capture, "operators").is_array())
    throw std::invalid_argument("Unsupported operator capture.");
  N dropped, errors;
  try {
    dropped = natural(get(capture, "dropped_events"));
    errors = natural(get(capture, "errors"));
  } catch (const std::exception&) {
    throw std::invalid_argument("Invalid capture loss or error counter.");
  }
  std::map<std::string, N> counts, modules;
  J skipped = J::object(), diagnostics = J::array();
  auto gap = [&](const std::string& name, const std::string& message,
                 const std::string& code = "unsupported_operator_formula") {
    diagnostics.push_back(
        {{"code", code}, {"severity", "warning"}, {"metric", "flops"}, {"operator", name}, {"message", message}});
  };
  if (dropped != 0 || errors != 0)
    gap("", "Capture lost " + dropped.str() + " events and recorded " + errors.str() + " collector errors.",
        "incomplete_capture");
  for (const auto& event : capture["operators"]) {
    if (!event.is_object() || !get(event, "operator").is_string() || !get(event, "overload").is_string() ||
        !get(event, "module").is_string() || !get(event, "inputs").is_array() || !get(event, "outputs").is_array() ||
        !get(event, "kwargs").is_object())
      throw std::invalid_argument("Invalid operator event.");
    auto name = event["operator"].get<std::string>();
    for (size_t pos = 0; (pos = name.find("::", pos)) != std::string::npos;) name.replace(pos, 2, ".");
    bool custom = event.contains("custom_flops") || event.contains("custom_error");
    auto ignored_rule = ignored().find(name);
    if (!custom && ignored_rule != ignored().end()) {
      auto& entry = skipped[name];
      if (entry.is_null()) entry = {{"calls", 0}, {"reason", ignored_rule->second}};
      entry["calls"] = entry["calls"].get<size_t>() + 1;
      continue;
    }
    auto packet = name.rfind("aten.", 0) == 0 ? name.substr(5) : "";
    auto rule = rules().find(packet);
    bool attention = rule != rules().end() && rule->second.kind == "_attention";
    try {
      Flags flags;
      if (packet == "sum" || packet == "mean" || packet == "cumsum" || packet == "cumsum_" ||
          packet == "linalg_vector_norm")
        flags.add(event["outputs"]);
      else if (packet == "pow" || packet == "pow_" || packet == "addcmul" || packet == "addcmul_" ||
               packet == "addcdiv" || packet == "addcdiv_")
        flags.add(event["inputs"]);
      else
        for (const auto* key : {"inputs", "kwargs", "outputs"}) flags.add(event[key]);
      if (event.contains("custom_error"))
        throw Unsupported(get(event, "custom_error").is_string() ? event["custom_error"].get<std::string>()
                                                                 : "Custom formula failed.");
      N count;
      if (custom)
        count = integer(event["custom_flops"]);
      else {
        if (rule == rules().end()) throw Uncounted();
        auto args = shapes(event["inputs"]), kw = J::object();
        for (auto it = event["kwargs"].begin(); it != event["kwargs"].end(); ++it) kw[it.key()] = shapes(*it);
        const auto& raw_out = event["outputs"];
        auto out = shapes(attention && !raw_out.empty() ? J::array({raw_out[0]}) : raw_out);
        if (out.empty()) throw Unsupported("Missing operator outputs.");
        bool scalar_power = false;
        if (packet == "pow" || packet == "pow_")
          for (const auto& v : event["inputs"])
            scalar_power |= v.is_number_float() || (v.is_object() && v.contains("nonfinite"));
        if (flags.tensor && !(flags.floating || flags.complex || scalar_power))
          count = 0;
        else {
          if (flags.complex && rule->second.kind != "_product" && rule->second.kind != "_convolution" && !attention)
            throw Unsupported("Supplemental FLOP formulas cover real arithmetic only.");
          if (!get(kw, "rounding_mode").is_null())
            throw Unsupported("Rounded division is not covered by the scalar arithmetic formula.");
          count = formula(packet, args, kw, out.size() == 1 ? out[0] : out);
        }
      }
      if (count < 0) throw Unsupported("Invalid formula result.");
      counts[name] += count;
      auto path = event["module"].get<std::string>();
      if (!path.empty()) {
        for (size_t pos = path.find('.'); pos != std::string::npos; pos = path.find('.', pos + 1))
          modules[path.substr(0, pos)] += count;
        modules[path] += count;
      }
      if (!custom && (flags.complex || attention))
        gap(name, "Shape counts cover only core arithmetic for complex inputs or fused attention.",
            "incomplete_operator_formula");
    } catch (const Uncounted&) {
      gap(name, name + " was observed 1 time(s), but no FLOP formula is registered.", "uncounted_operator");
    } catch (const std::exception& error) {
      gap(name, error.what());
    }
  }
  N total = 0;
  J by_operator = J::object(), by_module = J::object();
  for (const auto& item : counts) {
    total += item.second;
    by_operator[item.first] = number(item.second);
  }
  for (const auto& item : modules) by_module[item.first] = number(item.second);
  bool partial = !diagnostics.empty();
  return {{"schema_version", 1},
          {"context",
           {{"torch_version", context["torch_version"]},
            {"method", METHOD},
            {"counting_convention", "torchscan_flops_v1"},
            {"boundary", "dispatcher_operator"}}},
          {"total",
           {{"status", partial ? "partial" : "complete"},
            {"value", partial ? J(nullptr) : number(total)},
            {"known_value", number(total)},
            {"unit", "FLOPs"},
            {"scope", SCOPE},
            {"method", METHOD}}},
          {"by_operator", by_operator},
          {"by_module", by_module},
          {"ignored_operators", skipped},
          {"diagnostics", diagnostics}};
}
}  // namespace torchscan
