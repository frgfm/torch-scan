// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.

#include <ATen/core/TensorBody.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/record_function.h>
#include <Python.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/version.h>

#include <atomic>
#include <cmath>
#include <exception>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

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
  if (value.isDouble()) {
    auto number = value.toDouble();
    if (!std::isfinite(number)) return "{\"nonfinite\":" + quote(std::isnan(number) ? "nan" : (number < 0 ? "-inf" : "inf")) + "}";
    std::ostringstream out;
    out << std::setprecision(17) << std::showpoint << number;
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
  std::string name, overload, inputs, kwargs, outputs = "[]", module, custom;
};
struct Schema {
  std::vector<std::string> keywords;
  bool composite = false;
};
struct State {
  at::CallbackHandle handle = 0;
  std::thread::id owner = std::this_thread::get_id();
  size_t max_events, max_bytes, bytes = 0, dropped = 0;
  std::atomic<size_t> active{0}, errors{0};
  std::atomic<bool> closed{false};
  std::mutex mutex;
  std::vector<Event> events;
  std::vector<std::string> modules;
  std::unordered_map<uint64_t, std::unordered_map<int64_t, std::string>> sequence_modules;
  std::unordered_map<std::string, Schema> schemas;
  std::unordered_map<std::string, pybind11::object> formulas;
  std::unordered_set<std::string> composites;
  ~State() {
    if (!formulas.empty()) {
      pybind11::gil_scoped_acquire gil;
      formulas.clear();
    }
  }
};
std::mutex sessions_mutex;
std::unordered_map<uint64_t, std::weak_ptr<State>> sessions;
thread_local std::shared_ptr<State> current;
thread_local size_t depth = 0;
thread_local bool paused = false;
thread_local std::string backward_module;
struct Context : at::ObserverContext {
  std::shared_ptr<State> state;
  std::shared_ptr<State> previous_state;
  std::vector<c10::IValue> custom_inputs;
  std::vector<std::string> custom_keys;
  std::string custom_name;
  size_t index = SIZE_MAX;
  bool backward, transparent;
  int exceptions = std::uncaught_exceptions();
  std::string previous_module;
  explicit Context(std::shared_ptr<State> value, bool is_backward = false, bool is_transparent = false)
      : state(std::move(value)), backward(is_backward), transparent(is_transparent) {
    ++state->active;
    if (backward) {
      previous_state = current;
      current = state;
      previous_module = backward_module;
    } else if (!transparent) ++depth;
  }
  ~Context() override {
    if (backward) {
      backward_module = std::move(previous_module);
      current = std::move(previous_state);
    } else if (!transparent) --depth;
    if (std::uncaught_exceptions() > exceptions) ++state->errors;
    --state->active;
  }
};

std::unique_ptr<at::ObserverContext> begin(const at::RecordFunction& fn) {
  auto state = current;
  if (!state && fn.scope() == at::RecordScope::BACKWARD_FUNCTION) {
    // Autograd propagates callback handles to its workers. Resolve that session
    // once per backward node; unrelated threads have no capture callback.
    std::lock_guard<std::mutex> lock(sessions_mutex);
    for (const auto& callback : at::get_record_function_tls_().sorted_tls_callbacks_) {
      auto session = sessions.find(callback.handle_);
      if (session != sessions.end() && callback.enabled_) {
        state = session->second.lock();
        break;
      }
    }
  }
  if (!state || paused || state->closed) return nullptr;
  std::unique_ptr<Context> context;
  try {
    if (depth) return std::make_unique<Context>(state);
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closed) return nullptr;
    if (fn.scope() == at::RecordScope::BACKWARD_FUNCTION) {
      context = std::make_unique<Context>(state, true);
      backward_module.clear();
      auto thread = state->sequence_modules.find(fn.forwardThreadId());
      if (thread != state->sequence_modules.end()) {
        auto module = thread->second.find(fn.seqNr());
        if (module != thread->second.end()) backward_module = module->second;
      }
      return context;
    }
    auto op = fn.operator_name();
    if (!op) return nullptr;
    auto key = op->name + "." + op->overload_name;
    auto cached = state->schemas.find(key);
    Schema schema;
    if (cached == state->schemas.end()) {
      if (auto handle = c10::Dispatcher::singleton().findSchema(*op)) {
        schema.composite = handle->hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd);
      }
      if (auto definition = fn.operator_schema())
        for (const auto& arg : definition->arguments()) schema.keywords.push_back(arg.kwarg_only() ? arg.name() : "");
    } else
      schema = cached->second;
    if (schema.composite && state->composites.count(op->name) && !state->formulas.count(op->name)) {
      auto handle = c10::Dispatcher::singleton().findSchema(*op);
      bool decompose = true;
      for (const auto& input : fn.inputs())
        if (input.isTensor()) {
          const auto& tensor = input.toTensor();
          if (tensor.defined() && (tensor.layout() != c10::kStrided || tensor.is_nested() ||
              tensor.key_set().has(c10::DispatchKey::Python) ||
              handle->hasKernelForDispatchKey(c10::highestPriorityBackendTypeId(tensor.key_set()))))
            decompose = false;
        }
      if (decompose) return std::make_unique<Context>(state, false, true);
    }
    context = std::make_unique<Context>(state);
    if (state->events.size() >= state->max_events) {
      ++state->dropped;
      return context;
    }
    Event event;
    event.name = op->name;
    event.overload = op->overload_name;
    event.module = state->owner == std::this_thread::get_id() && !state->modules.empty() ? state->modules.back() : backward_module;
    if (state->formulas.count(event.name)) {
      context->custom_inputs.assign(fn.inputs().begin(), fn.inputs().end());
      context->custom_name = event.name;
      context->custom_keys = schema.keywords;
    }
    const auto& keys = schema.keywords;
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
    if (cached == state->schemas.end()) state->schemas.emplace(std::move(key), std::move(schema));
    if (fn.seqNr() >= 0 && !event.module.empty()) state->sequence_modules[fn.threadId()].emplace(fn.seqNr(), event.module);
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
  if (fn.outputs().empty() && fn.num_outputs()) {
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
    std::string custom_data;
    {
      auto custom = state.formulas.find(context->custom_name);
      if (custom != state.formulas.end()) {
        namespace py = pybind11;
        py::gil_scoped_acquire gil;
        try {
          const auto& keys = context->custom_keys;
          py::list positional;
          py::dict kwargs;
          size_t index = 0;
          for (const auto& input : context->custom_inputs) {
            auto value = torch::jit::toPyObject(input);
            if (index < keys.size() && !keys[index].empty())
              kwargs[py::str(keys[index])] = value;
            else
              positional.append(value);
            ++index;
          }
          py::tuple outputs(fn.outputs().size());
          for (size_t i = 0; i < fn.outputs().size(); ++i) outputs[i] = torch::jit::toPyObject(fn.outputs()[i]);
          if (outputs.size() == 1)
            kwargs["out_val"] = outputs[0];
          else
            kwargs["out_val"] = outputs;
          auto count = custom->second(*py::tuple(positional), **kwargs);
          if (PyFloat_Check(count.ptr())) {
            auto number = count.cast<double>();
            if (std::isfinite(number) && std::floor(number) == number) count = py::int_(count);
          }
          if (PyBool_Check(count.ptr())) throw std::runtime_error("Invalid custom FLOPs.");
          auto integer = PyNumber_Index(count.ptr());
          if (!integer) throw py::error_already_set();
          count = py::reinterpret_steal<py::object>(integer);
          if (PyObject_RichCompareBool(count.ptr(), py::int_(0).ptr(), Py_LT) != 0) throw std::runtime_error("Invalid custom FLOPs.");
          custom_data = ",\"custom_flops\":" + py::str(count).cast<std::string>();
        } catch (const std::exception& error) {
          custom_data = ",\"custom_error\":" + quote(std::string(error.what()).substr(0, 1024));
        }
      }
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.closed) return;
    auto& event = state.events[context->index];
    if (data.size() + custom_data.size() > state.max_bytes - state.bytes) {
      ++state.dropped;
      event = Event();
    } else {
      state.bytes += data.size() + custom_data.size();
      event.outputs = std::move(data);
      event.custom = std::move(custom_data);
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
void deactivate(const std::shared_ptr<State>& state) {
  state->closed = true;
  at::removeCallback(state->handle);
  std::lock_guard<std::mutex> lock(sessions_mutex);
  sessions.erase(state->handle);
}
void destroy(PyObject* capsule) {
  auto state = static_cast<std::shared_ptr<State>*>(PyCapsule_GetPointer(capsule, "torchscan.capture"));
  if (state && state->get() == current.get()) {
    deactivate(*state);
    current.reset();
  }
  delete state;
}
PyObject* start(PyObject*, PyObject* args) {
  unsigned long long events, bytes;
  PyObject* formulas = nullptr;
  PyObject* composites = nullptr;
  if (!PyArg_ParseTuple(args, "KK|O!O", &events, &bytes, &PyDict_Type, &formulas, &composites)) return nullptr;
  if (current) {
    PyErr_SetString(PyExc_RuntimeError, "A native capture is already active on this thread.");
    return nullptr;
  }
  try {
    auto state = std::make_shared<State>();
    state->max_events = events;
    state->max_bytes = bytes;
    if (formulas)
      for (auto item : pybind11::reinterpret_borrow<pybind11::dict>(formulas))
        state->formulas.emplace(pybind11::cast<std::string>(item.first), pybind11::reinterpret_borrow<pybind11::object>(item.second));
    if (composites)
      for (auto item : pybind11::reinterpret_borrow<pybind11::iterable>(composites))
        state->composites.emplace(pybind11::cast<std::string>(item));
    state->handle = at::addThreadLocalCallback(at::RecordFunctionCallback(&begin, &end)
                                                   .needsInputs(true)
                                                   .needsOutputs(true)
                                                   .scopes({at::RecordScope::FUNCTION, at::RecordScope::BACKWARD_FUNCTION}));
    current = state;
    {
      std::lock_guard<std::mutex> lock(sessions_mutex);
      sessions[state->handle] = state;
    }
    auto holder = std::make_unique<std::shared_ptr<State>>(state);
    auto capsule = PyCapsule_New(holder.get(), "torchscan.capture", &destroy);
    if (!capsule) {
      deactivate(state);
      current.reset();
    } else
      holder.release();
    return capsule;
  } catch (const std::exception& e) {
    if (current) deactivate(current);
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
    deactivate(state);
    current.reset();
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->active) ++state->errors;
    std::ostringstream out;
    out << "{\"dropped_events\":" << state->dropped << ",\"errors\":" << state->errors << ",\"operators\":[";
    bool first = true;
    for (const auto& event : state->events) {
      if (event.name.empty()) continue;
      if (!first) out << ',';
      first = false;
      out << "{\"operator\":" << quote(event.name) << ",\"overload\":" << quote(event.overload)
          << ",\"inputs\":" << event.inputs << ",\"kwargs\":" << event.kwargs << ",\"outputs\":" << event.outputs
          << ",\"module\":" << quote(event.module) << event.custom << '}';
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
PyObject* pause(PyObject*, PyObject* value) {
  int requested = PyObject_IsTrue(value);
  if (requested < 0) return nullptr;
  bool previous = current && paused;
  if (current) paused = requested;
  return PyBool_FromLong(previous);
}
PyMethodDef methods[] = {{"start", start, METH_VARARGS, nullptr},
                         {"stop", stop, METH_O, nullptr},
                         {"module", module, METH_VARARGS, nullptr},
                         {"pause", pause, METH_O, nullptr},
                         {nullptr, nullptr, 0, nullptr}};
PyModuleDef definition = {PyModuleDef_HEAD_INIT, "_torchscan_native", nullptr, -1, methods};
}  // namespace
PyMODINIT_FUNC PyInit__torchscan_native() {
  auto module = PyModule_Create(&definition);
  if (module) PyModule_AddStringConstant(module, "torch_version", TORCH_VERSION);
  return module;
}
