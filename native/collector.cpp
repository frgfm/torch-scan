// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.

#include <ATen/core/TensorBody.h>
#include <ATen/core/function_schema.h>
#include <ATen/record_function.h>
#include <Python.h>
#include <torch/version.h>

#include <atomic>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace {
std::string quote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}

// Copy metadata, never IValues or tensors: capture must not extend tensor lifetimes.
std::string metadata(const c10::IValue& value, size_t& budget, bool output = false, size_t depth = 0) {
  if (!budget || depth > 8) return "{\"unknown\":\"metadata limit\"}";
  --budget;
  if (value.isNone()) return "null";
  if (value.isTensor()) {
    const auto& tensor = value.toTensor();
    if (!tensor.defined()) return "null";
    if (tensor.key_set().has(c10::DispatchKey::Python) || tensor.layout() != c10::kStrided || tensor.is_nested())
      return "{\"unknown\":\"tensor layout or subclass\"}";
    std::ostringstream out;
    out << "{\"shape\":[";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
      if (i) out << ',';
      out << tensor.size(i);
    }
    out << "],\"dtype\":" << quote(c10::toString(tensor.scalar_type()))
        << ",\"device\":" << quote(tensor.device().str())
        << ",\"floating\":" << (tensor.is_floating_point() ? "true" : "false")
        << ",\"complex\":" << (tensor.is_complex() ? "true" : "false") << '}';
    return out.str();
  }
  if (value.isList() || value.isTuple()) {
    std::ostringstream out;
    out << '[';
    const auto values = value.isTuple() ? value.toTupleRef().elements() : value.toListRef();
    if (values.size() > 256) return "{\"unknown\":\"container limit\"}";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) out << ',';
      out << metadata(values[i], budget, output, depth + 1);
    }
    out << ']';
    return out.str();
  }
  if (output) return "{\"unknown\":\"non-tensor output\"}";
  if (value.isBool()) return value.toBool() ? "true" : "false";
  if (value.isInt()) return std::to_string(value.toInt());
  if (value.isDouble() && std::isfinite(value.toDouble())) {
    std::ostringstream out;
    out << std::setprecision(17) << value.toDouble();
    return out.str();
  }
  // Preserve formula options, not arbitrary strings supplied to custom operators.
  if (value.isString()) {
    auto text = value.toStringRef();
    if (text == "none" || text == "tanh" || text == "floor" || text == "trunc") return quote(text);
  }
  return "{\"unknown\":\"argument type\"}";
}

struct Event {
  std::string name, overload, inputs, kwargs, outputs = "[]", module;
};
struct State {
  at::CallbackHandle handle = 0;
  std::thread::id owner = std::this_thread::get_id();
  size_t max_events, max_bytes, bytes = 0, dropped = 0;
  std::atomic<size_t> depth{0}, errors{0};
  std::vector<Event> events;
  std::vector<std::string> modules;
  std::unordered_map<std::string, std::vector<std::string>> keywords;
};
thread_local std::shared_ptr<State> current;
struct Context : at::ObserverContext {
  std::shared_ptr<State> state;
  size_t index = SIZE_MAX;
  explicit Context(std::shared_ptr<State> value) : state(std::move(value)) { ++state->depth; }
  ~Context() override { --state->depth; }
};

std::unique_ptr<at::ObserverContext> begin(const at::RecordFunction& fn) {
  auto state = current;
  if (!state) return nullptr;
  std::unique_ptr<Context> context;
  try {
    auto op = fn.operator_name();
    if (!op) return nullptr;
    context = std::make_unique<Context>(state);
    if (state->depth != 1) return context;  // Outermost native operators own their internal calls.
    if (state->events.size() >= state->max_events) {
      ++state->dropped;
      return context;
    }
    Event event;
    event.name = op->name;
    event.overload = op->overload_name;
    event.module = state->modules.empty() ? "" : state->modules.back();
    auto key = event.name + "." + event.overload;
    if (!state->keywords.count(key)) {
      std::vector<std::string> keys;
      if (auto schema = fn.operator_schema())
        for (const auto& arg : schema->arguments()) keys.push_back(arg.kwarg_only() ? arg.name() : "");
      state->keywords.emplace(key, std::move(keys));
    }
    const auto& keys = state->keywords.at(key);
    std::ostringstream inputs, kwargs;
    inputs << '[';
    kwargs << '{';
    size_t i = 0, positional = 0, named = 0, budget = 4096;
    for (const auto& value : fn.inputs()) {
      auto data = metadata(value, budget);
      if (i < keys.size() && !keys[i].empty()) {
        if (named++) kwargs << ',';
        kwargs << quote(keys[i]) << ':' << data;
      } else {
        if (positional++) inputs << ',';
        inputs << data;
      }
      ++i;
    }
    inputs << ']';
    kwargs << '}';
    event.inputs = inputs.str();
    event.kwargs = kwargs.str();
    size_t bytes =
        event.name.size() + event.overload.size() + event.module.size() + event.inputs.size() + event.kwargs.size();
    if (bytes > state->max_bytes - state->bytes) {
      ++state->dropped;
      return context;
    }
    state->events.push_back(std::move(event));
    state->bytes += bytes;
    context->index = state->events.size() - 1;
  } catch (...) {
    ++state->errors;
  }
  return context;
}

void end(const at::RecordFunction& fn, at::ObserverContext* raw) {
  auto context = static_cast<Context*>(raw);
  if (!context || context->index == SIZE_MAX) return;
  auto& state = *context->state;
  if (state.owner != std::this_thread::get_id()) {
    ++state.errors;
    return;
  }
  try {
    std::ostringstream out;
    out << '[';
    size_t budget = 4096, i = 0;
    for (const auto& value : fn.outputs()) {
      if (i++) out << ',';
      out << metadata(value, budget, true);
    }
    out << ']';
    auto data = out.str();
    if (data.size() > state.max_bytes - state.bytes) {
      ++state.dropped;
      state.events.pop_back();
    } else {
      state.bytes += data.size();
      state.events[context->index].outputs = std::move(data);
    }
  } catch (...) {
    ++state.errors;
  }
}

std::shared_ptr<State>* state_from(PyObject* capsule) {
  auto state = static_cast<std::shared_ptr<State>*>(PyCapsule_GetPointer(capsule, "torchscan.capture"));
  if (state && state->get() != current.get()) {
    PyErr_SetString(PyExc_RuntimeError, "Capture is inactive or belongs to another thread.");
    return nullptr;
  }
  return state;
}
void destroy(PyObject* capsule) {
  auto state = static_cast<std::shared_ptr<State>*>(PyCapsule_GetPointer(capsule, "torchscan.capture"));
  if (state && state->get() == current.get()) {
    at::removeCallback((*state)->handle);
    current.reset();
  }
  delete state;
}
PyObject* start(PyObject*, PyObject* args) {
  unsigned long long events, bytes;
  if (!PyArg_ParseTuple(args, "KK", &events, &bytes)) return nullptr;
  if (current) {
    PyErr_SetString(PyExc_RuntimeError, "A native capture is already active on this thread.");
    return nullptr;
  }
  try {
    auto state = std::make_shared<State>();
    state->max_events = events;
    state->max_bytes = bytes;
    state->handle = at::addThreadLocalCallback(at::RecordFunctionCallback(&begin, &end)
                                                   .needsInputs(true)
                                                   .needsOutputs(true)
                                                   .scopes({at::RecordScope::FUNCTION}));
    current = state;
    auto holder = std::make_unique<std::shared_ptr<State>>(state);
    auto capsule = PyCapsule_New(holder.get(), "torchscan.capture", &destroy);
    if (!capsule) {
      at::removeCallback(state->handle);
      current.reset();
    } else
      holder.release();
    return capsule;
  } catch (const std::exception& e) {
    if (current) at::removeCallback(current->handle);
    current.reset();
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}
PyObject* stop(PyObject*, PyObject* capsule) {
  try {
    auto pointer = state_from(capsule);
    if (!pointer) return nullptr;
    auto state = *pointer;
    at::removeCallback(state->handle);
    current.reset();
    if (state->depth) ++state->errors;
    std::ostringstream out;
    out << "{\"dropped_events\":" << state->dropped << ",\"errors\":" << state->errors << ",\"operators\":[";
    for (size_t i = 0; i < state->events.size(); ++i) {
      const auto& event = state->events[i];
      if (i) out << ',';
      out << "{\"operator\":" << quote(event.name) << ",\"overload\":" << quote(event.overload)
          << ",\"inputs\":" << event.inputs << ",\"kwargs\":" << event.kwargs << ",\"outputs\":" << event.outputs
          << ",\"module\":" << quote(event.module) << '}';
    }
    out << "]}";
    auto data = out.str();
    return PyUnicode_FromStringAndSize(data.data(), data.size());
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}
PyObject* module(PyObject*, PyObject* args) {
  PyObject* capsule;
  const char* name = nullptr;
  if (!PyArg_ParseTuple(args, "O|s", &capsule, &name)) return nullptr;
  auto state = state_from(capsule);
  if (!state) return nullptr;
  try {
    if (name)
      (*state)->modules.emplace_back(name);
    else if (!(*state)->modules.empty())
      (*state)->modules.pop_back();
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}
PyMethodDef methods[] = {{"start", start, METH_VARARGS, nullptr},
                         {"stop", stop, METH_O, nullptr},
                         {"module", module, METH_VARARGS, nullptr},
                         {nullptr, nullptr, 0, nullptr}};
PyModuleDef definition = {PyModuleDef_HEAD_INIT, "_torchscan_native", nullptr, -1, methods};
}  // namespace
PyMODINIT_FUNC PyInit__torchscan_native() {
  auto module = PyModule_Create(&definition);
  if (module) PyModule_AddStringConstant(module, "torch_version", TORCH_VERSION);
  return module;
}
