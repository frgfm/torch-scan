// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.

use super::*;
use pyo3::{
    exceptions::{PyKeyError, PyNotImplementedError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
    IntoPyObjectExt,
};

fn input(value: &Bound<'_, PyAny>, tuples: bool, depth: usize) -> PyResult<Value> {
    if depth > 128 {
        return Err(PyValueError::new_err("Capture nesting exceeds 128 levels."));
    }
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(v) = value.cast::<PyBool>() {
        return Ok(Value::Bool(v.is_true()));
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(v) = value.extract::<i64>() {
            return Ok(json!(v));
        }
        if let Ok(v) = value.extract::<u64>() {
            return Ok(json!(v));
        }
        return Number::from_str(value.str()?.to_str()?)
            .map(Value::Number)
            .map_err(|_| PyValueError::new_err("Invalid integer."));
    }
    if let Ok(v) = value.cast::<PyFloat>() {
        return Number::from_f64(v.value())
            .map(Value::Number)
            .ok_or_else(|| {
                PyValueError::new_err("Nonfinite numbers require capture scalar metadata.")
            });
    }
    if let Ok(v) = value.cast::<PyString>() {
        return Ok(Value::String(v.to_str()?.into()));
    }
    if let Ok(v) = value.cast::<PyList>() {
        return v
            .iter()
            .map(|v| input(&v, tuples, depth + 1))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    if tuples {
        if let Ok(v) = value.cast::<PyTuple>() {
            return v
                .iter()
                .map(|v| input(&v, tuples, depth + 1))
                .collect::<PyResult<Vec<_>>>()
                .map(Value::Array);
        }
    }
    if let Ok(v) = value.cast::<PyDict>() {
        return v
            .iter()
            .map(|(key, v)| {
                let key = key
                    .extract::<String>()
                    .map_err(|_| PyValueError::new_err("Capture object keys must be strings."))?;
                Ok((key, input(&v, tuples, depth + 1)?))
            })
            .collect::<PyResult<Map<_, _>>>()
            .map(Value::Object);
    }
    Err(PyValueError::new_err(
        "Capture must contain JSON values only.",
    ))
}
fn output(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(v) => v.into_py_any(py),
        Value::String(v) => v.into_py_any(py),
        Value::Number(v) => {
            if let Some(v) = v.as_i64() {
                return v.into_py_any(py);
            }
            if let Some(v) = v.as_u64() {
                return v.into_py_any(py);
            }
            if v.as_str().contains(['.', 'e', 'E']) {
                return v
                    .as_f64()
                    .ok_or_else(|| PyValueError::new_err("Invalid number."))
                    .and_then(|v| v.into_py_any(py));
            }
            BigInt::from_str(v.as_str())
                .map_err(|_| PyValueError::new_err("Invalid integer."))?
                .into_py_any(py)
        }
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(output(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(items) => {
            let dict = PyDict::new(py);
            for (key, value) in items {
                dict.set_item(key, output(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}
fn error(error: Error) -> PyErr {
    match error {
        Error::Invalid(message) => PyValueError::new_err(message),
        Error::Unsupported(message) => PyNotImplementedError::new_err(message),
        Error::Uncounted => PyKeyError::new_err("No registered formula."),
    }
}
#[pyfunction(name = "analyze")]
fn analyze_py(py: Python<'_>, capture: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let capture = input(capture, false, 0)?;
    let report = py.detach(|| analyze(&capture)).map_err(error)?;
    output(py, report)
}
#[pyfunction(name = "formula")]
fn formula_py(
    py: Python<'_>,
    name: &str,
    args: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyAny>,
    out_shape: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let args = input(args, true, 0)?;
    let kwargs = input(kwargs, true, 0)?;
    let out = input(out_shape, true, 0)?;
    let count = formula(
        name,
        array(&args).map_err(error)?,
        kwargs
            .as_object()
            .ok_or_else(|| PyValueError::new_err("Expected keyword arguments."))?,
        &out,
    )
    .map_err(error)?;
    count.into_py_any(py)
}
#[pyfunction]
fn formula_names() -> Vec<String> {
    rules().keys().cloned().collect()
}
#[pymodule]
fn _analysis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_py, m)?)?;
    m.add_function(wrap_pyfunction!(formula_py, m)?)?;
    m.add_function(wrap_pyfunction!(formula_names, m)?)?;
    Ok(())
}
