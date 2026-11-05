// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.
#include <Python.h>

#include <memory>

#include "count.hpp"

namespace {
using namespace torchscan;
struct Decref {
  void operator()(PyObject* value) const { Py_DECREF(value); }
};
using PyRef = std::unique_ptr<PyObject, Decref>;
PyObject* checked(PyObject* value) {
  if (!value) throw std::runtime_error("Python value conversion failed.");
  return value;
}
std::string text(PyObject* value) {
  Py_ssize_t length;
  const char* data = PyUnicode_AsUTF8AndSize(value, &length);
  if (!data) throw std::invalid_argument("Capture strings must be valid Unicode.");
  return {data, static_cast<size_t>(length)};
}
J input(PyObject* value, bool tuples = false, size_t depth = 0) {
  if (depth > 128) throw std::invalid_argument("Capture nesting exceeds 128 levels.");
  if (value == Py_None) return nullptr;
  if (PyBool_Check(value)) return value == Py_True;
  if (PyLong_Check(value)) {
    int overflow;
    auto n = PyLong_AsLongLongAndOverflow(value, &overflow);
    if (!overflow && !PyErr_Occurred()) return n;
    PyErr_Clear();
    auto u = PyLong_AsUnsignedLongLong(value);
    if (!PyErr_Occurred()) return u;
    PyErr_Clear();
    PyRef decimal(checked(PyObject_Str(value)));
    return number(N(text(decimal.get())));
  }
  if (PyFloat_Check(value)) {
    double n = PyFloat_AsDouble(value);
    if (!std::isfinite(n)) throw std::invalid_argument("Nonfinite numbers require capture scalar metadata.");
    return n;
  }
  if (PyUnicode_Check(value)) return text(value);
  if (PyList_Check(value) || (tuples && PyTuple_Check(value))) {
    J result = J::array();
    bool list = PyList_Check(value);
    Py_ssize_t count = list ? PyList_GET_SIZE(value) : PyTuple_GET_SIZE(value);
    result.get_ref<J::array_t&>().reserve(count);
    for (Py_ssize_t i = 0; i < count; ++i)
      result.push_back(input(list ? PyList_GET_ITEM(value, i) : PyTuple_GET_ITEM(value, i), tuples, depth + 1));
    return result;
  }
  if (PyDict_Check(value)) {
    J result = J::object();
    Py_ssize_t pos = 0;
    PyObject *key, *item;
    while (PyDict_Next(value, &pos, &key, &item)) {
      if (!PyUnicode_Check(key)) throw std::invalid_argument("Capture object keys must be strings.");
      result[text(key)] = input(item, tuples, depth + 1);
    }
    return result;
  }
  throw std::invalid_argument("Capture must contain JSON values only.");
}
PyObject* output(const J& value) {
  if (value.is_null()) {
    Py_RETURN_NONE;
  }
  if (value.is_boolean()) return PyBool_FromLong(value.get<bool>());
  if (value.is_number_unsigned()) return PyLong_FromUnsignedLongLong(value.get<uint64_t>());
  if (value.is_number_integer()) return PyLong_FromLongLong(value.get<int64_t>());
  if (value.is_binary()) {
    auto decimal = integer(value).str();
    return PyLong_FromString(decimal.data(), nullptr, 10);
  }
  if (value.is_number_float()) return PyFloat_FromDouble(value.get<double>());
  if (value.is_string()) {
    const auto& s = value.get_ref<const std::string&>();
    return PyUnicode_FromStringAndSize(s.data(), s.size());
  }
  if (value.is_array()) {
    PyRef result(checked(PyList_New(value.size())));
    for (size_t i = 0; i < value.size(); ++i) PyList_SET_ITEM(result.get(), i, checked(output(value[i])));
    return result.release();
  }
  PyRef result(checked(PyDict_New()));
  for (auto it = value.begin(); it != value.end(); ++it) {
    PyRef key(checked(PyUnicode_FromStringAndSize(it.key().data(), it.key().size()))), item(checked(output(*it)));
    if (PyDict_SetItem(result.get(), key.get(), item.get()) < 0)
      throw std::runtime_error("Python report conversion failed.");
  }
  return result.release();
}
PyObject* failure(PyObject* type, const std::exception& error) {
  PyErr_Clear();
  PyErr_SetString(type, error.what());
  return nullptr;
}
PyObject* py_analyze(PyObject*, PyObject* capture) {
  try {
    auto data = input(capture);
    J report;
    {
      struct Unlock {
        PyThreadState* state = PyEval_SaveThread();
        ~Unlock() { PyEval_RestoreThread(state); }
      } unlock;
      report = analyze(data);
    }
    return output(report);
  } catch (const std::exception& error) {
    return failure(PyExc_ValueError, error);
  }
}
PyObject* py_formula(PyObject*, PyObject* values) {
  const char* name;
  PyObject *args, *kwargs, *out;
  if (!PyArg_ParseTuple(values, "sOOO", &name, &args, &kwargs, &out)) return nullptr;
  try {
    return output(number(formula(name, input(args, true), input(kwargs, true), input(out, true))));
  } catch (const Uncounted& error) {
    return failure(PyExc_KeyError, error);
  } catch (const Unsupported& error) {
    return failure(PyExc_NotImplementedError, error);
  } catch (const std::exception& error) {
    return failure(PyExc_ValueError, error);
  }
}
PyObject* names(PyObject*, PyObject*) {
  try {
    J result = J::array();
    for (const auto& item : rules()) result.push_back(item.first);
    return output(result);
  } catch (const std::exception& error) {
    return failure(PyExc_ValueError, error);
  }
}
PyMethodDef methods[] = {{"analyze", py_analyze, METH_O, nullptr},
                         {"formula", py_formula, METH_VARARGS, nullptr},
                         {"formula_names", names, METH_NOARGS, nullptr},
                         {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_analysis", nullptr, -1, methods, nullptr, nullptr, nullptr, nullptr};
}  // namespace
PyMODINIT_FUNC PyInit__analysis() { return PyModule_Create(&module); }
