// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.

mod formulas;
#[cfg(feature = "python")]
mod python;

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde_json::{json, Map, Number, Value};
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::OnceLock;

pub const METHOD: &str = "ATen.RecordFunction/torchscan_flops_v1";
pub const SCOPE: &str = "calling_thread_and_autograd_eager_ops";
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Unsupported(String),
    Uncounted,
}
impl std::fmt::Display for Error {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Unsupported(message) => out.write_str(message),
            Self::Uncounted => out.write_str("No registered formula."),
        }
    }
}
impl std::error::Error for Error {}
fn unsupported(message: &str) -> Error {
    Error::Unsupported(message.into())
}
fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

fn integer(value: &Value) -> Result<BigInt> {
    let Value::Number(n) = value else {
        return Err(unsupported("Expected an integer."));
    };
    if let Some(n) = n.as_i64() {
        return Ok(n.into());
    }
    if let Some(n) = n.as_u64() {
        return Ok(n.into());
    }
    BigInt::from_str(n.as_str()).map_err(|_| unsupported("Expected an integer."))
}
fn natural(value: &Value) -> Result<BigInt> {
    let n = integer(value)?;
    if n.is_negative() {
        Err(unsupported("Expected a nonnegative integer."))
    } else {
        Ok(n)
    }
}
fn number(n: BigInt) -> Value {
    Value::Number(
        Number::from_str(&n.to_string()).expect("BigInt always formats as a JSON integer"),
    )
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| unsupported("Expected shape or argument array."))
}
fn product(value: &Value) -> Result<BigInt> {
    array(value)?
        .iter()
        .try_fold(BigInt::one(), |total, n| Ok(total * natural(n)?))
}
fn product_slice(values: &[Value]) -> Result<BigInt> {
    values
        .iter()
        .try_fold(BigInt::one(), |total, n| Ok(total * natural(n)?))
}
fn scalar(value: &Value, default: f64) -> Result<f64> {
    match value {
        Value::Null => Ok(default),
        Value::Bool(b) => Ok(u8::from(*b).into()),
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| unsupported("Unsupported numeric option.")),
        Value::Object(m) if m.len() == 1 => match m.get("nonfinite").and_then(Value::as_str) {
            Some("inf") => Ok(f64::INFINITY),
            Some("-inf") => Ok(f64::NEG_INFINITY),
            Some("nan") => Ok(f64::NAN),
            _ => Err(unsupported("Invalid scalar metadata.")),
        },
        _ => Err(unsupported("Unsupported scalar option.")),
    }
}
fn boolean(value: &Value, default: bool) -> Result<bool> {
    if value.is_null() {
        Ok(default)
    } else {
        value
            .as_bool()
            .ok_or_else(|| unsupported("Expected a boolean option."))
    }
}
fn arg<'a>(
    args: &'a [Value],
    kwargs: &'a Map<String, Value>,
    index: usize,
    name: &str,
) -> &'a Value {
    args.get(index)
        .or_else(|| kwargs.get(name))
        .unwrap_or(&Value::Null)
}
fn axis(shape: &[Value], dim: &Value, default: i64) -> Result<usize> {
    let d = if dim.is_null() {
        default
    } else {
        integer(dim)?
            .to_i64()
            .ok_or_else(|| unsupported("Invalid dimension."))?
    };
    let rank = shape.len() as i64;
    let d = if d < 0 { d + rank } else { d };
    if d < 0 || d >= rank {
        Err(unsupported("Invalid dimension."))
    } else {
        Ok(d as usize)
    }
}

struct Rule {
    kind: String,
    params: Map<String, Value>,
}
fn rules() -> &'static BTreeMap<String, Rule> {
    static RULES: OnceLock<BTreeMap<String, Rule>> = OnceLock::new();
    RULES.get_or_init(|| {
        let groups: Vec<(String, Map<String, Value>, String)> =
            serde_json::from_str(include_str!("../rules.json")).expect("Valid embedded rules");
        let mut result = BTreeMap::new();
        for (kind, params, names) in groups {
            for name in names.split_whitespace() {
                result.insert(
                    name.into(),
                    Rule {
                        kind: kind.clone(),
                        params: params.clone(),
                    },
                );
            }
        }
        result
    })
}
fn ignored() -> &'static BTreeMap<String, String> {
    static IGNORED: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    IGNORED.get_or_init(|| {
        let groups: Vec<(String, String)> = serde_json::from_str(include_str!("../ignored.json"))
            .expect("Valid embedded ignored operators");
        let mut result = BTreeMap::new();
        for (reason, names) in groups {
            for name in names.split_whitespace() {
                result.insert(name.into(), reason.clone());
            }
        }
        result
    })
}
pub fn formula(
    name: &str,
    args: &[Value],
    kwargs: &Map<String, Value>,
    output: &Value,
) -> Result<BigInt> {
    let rule = rules().get(name).ok_or(Error::Uncounted)?;
    formulas::evaluate(&rule.kind, &rule.params, args, kwargs, output)
}

// Turn tensor metadata into shapes while preserving scalar formula options.
fn shapes(value: &Value) -> Result<Value> {
    match value {
        Value::Object(m) => {
            if m.len() == 1
                && m.get("nonfinite")
                    .and_then(Value::as_str)
                    .is_some_and(|v| matches!(v, "inf" | "-inf" | "nan"))
            {
                return Ok(value.clone());
            }
            if let Some(reason) = m.get("unknown") {
                return Err(unsupported(
                    reason.as_str().unwrap_or("Unknown argument metadata."),
                ));
            }
            let dims = array(&value["shape"])?;
            if !value["floating"].is_boolean()
                || !value["complex"].is_boolean()
                || !value["dtype"].is_string()
                || !value["device"].is_string()
            {
                return Err(unsupported("Invalid tensor metadata."));
            }
            for dim in dims {
                if natural(dim)?.to_i64().is_none() {
                    return Err(unsupported("Invalid tensor dimension."));
                }
            }
            Ok(value["shape"].clone())
        }
        Value::Array(items) => items
            .iter()
            .map(shapes)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        _ => Ok(value.clone()),
    }
}
fn tensor_flags(value: &Value) -> Result<(bool, bool, bool)> {
    if value.is_object() && value.get("shape").is_some() {
        return Ok((
            true,
            value["floating"]
                .as_bool()
                .ok_or_else(|| unsupported("Invalid tensor metadata."))?,
            value["complex"]
                .as_bool()
                .ok_or_else(|| unsupported("Invalid tensor metadata."))?,
        ));
    }
    let mut result = (false, false, false);
    let values: Box<dyn Iterator<Item = &Value> + '_> = match value {
        Value::Array(items) => Box::new(items.iter()),
        Value::Object(items) => Box::new(items.values()),
        _ => return Ok(result),
    };
    for item in values {
        let v = tensor_flags(item)?;
        result = (result.0 || v.0, result.1 || v.1, result.2 || v.2);
    }
    Ok(result)
}
fn core_product(packet: &str) -> bool {
    matches!(packet, "mm" | "bmm" | "addmm" | "baddbmm" | "_scaled_mm")
}
fn core_conv(packet: &str) -> bool {
    matches!(
        packet,
        "convolution"
            | "_convolution"
            | "cudnn_convolution"
            | "convolution_overrideable"
            | "_slow_conv2d_forward"
    )
}
fn core_attention(packet: &str) -> bool {
    rules().get(packet).is_some_and(|r| r.kind == "_attention")
}

pub fn analyze(capture: &Value) -> Result<Value> {
    if capture["schema_version"].as_u64() != Some(1)
        || capture["report_type"] != "operator_capture"
        || capture["context"]["method"] != METHOD
        || capture["context"]["scope"] != SCOPE
        || capture["context"]["boundary"] != "dispatcher_operator"
        || !capture["context"]["torch_version"].is_string()
        || !capture["inputs"].is_object()
        || !capture["operators"].is_array()
    {
        return Err(invalid("Unsupported operator capture."));
    }
    let dropped = natural(&capture["dropped_events"])
        .map_err(|_| invalid("Invalid capture loss counter."))?;
    let errors =
        natural(&capture["errors"]).map_err(|_| invalid("Invalid capture error counter."))?;
    let mut counts: BTreeMap<String, BigInt> = BTreeMap::new();
    let mut modules: BTreeMap<String, BigInt> = BTreeMap::new();
    let mut skipped = Map::new();
    let mut diagnostics = Vec::new();
    let mut gap = |operator: &str, message: String, code: &str| {
        diagnostics.push(json!({"code":code,"severity":"warning","metric":"flops","operator":operator,"message":message}))
    };
    if !dropped.is_zero() || !errors.is_zero() {
        gap(
            "",
            format!("Capture lost {dropped} events and recorded {errors} collector errors."),
            "incomplete_capture",
        );
    }
    for event in capture["operators"]
        .as_array()
        .expect("Validated operators")
    {
        if !event.is_object()
            || !event["operator"].is_string()
            || !event["overload"].is_string()
            || !event["module"].is_string()
            || !event["inputs"].is_array()
            || !event["outputs"].is_array()
            || !event["kwargs"].is_object()
        {
            return Err(invalid("Invalid operator event."));
        }
        let name = event["operator"]
            .as_str()
            .expect("Validated operator")
            .replace("::", ".");
        let custom = event.get("custom_flops").is_some() || event.get("custom_error").is_some();
        if let Some(reason) = ignored().get(&name).filter(|_| !custom) {
            let entry = skipped
                .entry(name)
                .or_insert_with(|| json!({"calls":0,"reason":reason}));
            entry["calls"] = json!(entry["calls"].as_u64().expect("Call count") + 1);
            continue;
        }
        let packet = name.strip_prefix("aten.").unwrap_or("");
        let result = (|| -> Result<(BigInt, bool)> {
            let flags = if matches!(
                packet,
                "sum" | "mean" | "cumsum" | "cumsum_" | "linalg_vector_norm"
            ) {
                tensor_flags(&event["outputs"])?
            } else if matches!(
                packet,
                "pow" | "pow_" | "addcmul" | "addcmul_" | "addcdiv" | "addcdiv_"
            ) {
                tensor_flags(&event["inputs"])?
            } else {
                let mut flags = (false, false, false);
                for key in ["inputs", "kwargs", "outputs"] {
                    let item = tensor_flags(&event[key])?;
                    flags = (flags.0 || item.0, flags.1 || item.1, flags.2 || item.2);
                }
                flags
            };
            let complex = flags.2;
            if let Some(error) = event.get("custom_error") {
                return Err(unsupported(
                    error.as_str().unwrap_or("Custom formula failed."),
                ));
            }
            let count = if custom {
                integer(&event["custom_flops"])?
            } else {
                if !rules().contains_key(packet) {
                    return Err(Error::Uncounted);
                }
                let args = shapes(&event["inputs"])?;
                let kwargs = event["kwargs"]
                    .as_object()
                    .expect("Validated kwargs")
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), shapes(v)?)))
                    .collect::<Result<Map<_, _>>>()?;
                let raw_out = if core_attention(packet) {
                    Value::Array(
                        event["outputs"]
                            .as_array()
                            .expect("Outputs")
                            .iter()
                            .take(1)
                            .cloned()
                            .collect(),
                    )
                } else {
                    event["outputs"].clone()
                };
                let out = shapes(&raw_out)?;
                let out = array(&out)?;
                if out.is_empty() {
                    return Err(unsupported("Missing operator outputs."));
                }
                let output = if out.len() == 1 {
                    &out[0]
                } else {
                    &Value::Array(out.to_vec())
                };
                let scalar_power = matches!(packet, "pow" | "pow_")
                    && event["inputs"].as_array().expect("Inputs").iter().any(|v| {
                        v.get("nonfinite").is_some()
                            || v.as_number()
                                .is_some_and(|n| n.to_string().contains(['.', 'e', 'E']))
                    });
                if flags.0 && !(flags.1 || flags.2 || scalar_power) {
                    BigInt::zero()
                } else {
                    if complex
                        && !(core_product(packet) || core_conv(packet) || core_attention(packet))
                    {
                        return Err(unsupported(
                            "Supplemental FLOP formulas cover real arithmetic only.",
                        ));
                    }
                    if !kwargs
                        .get("rounding_mode")
                        .unwrap_or(&Value::Null)
                        .is_null()
                    {
                        return Err(unsupported(
                            "Rounded division is not covered by the scalar arithmetic formula.",
                        ));
                    }
                    formula(packet, array(&args)?, &kwargs, output)?
                }
            };
            if count.is_negative() {
                return Err(unsupported("Invalid formula result."));
            }
            Ok((count, !custom && (complex || core_attention(packet))))
        })();
        match result {
            Ok((count, incomplete)) => {
                *counts.entry(name.clone()).or_default() += &count;
                let path = event["module"].as_str().expect("Validated module");
                if !path.is_empty() {
                    for (pos, _) in path.match_indices('.') {
                        *modules.entry(path[..pos].into()).or_default() += &count;
                    }
                    *modules.entry(path.into()).or_default() += &count;
                }
                if incomplete {
                    gap(&name, "Shape counts cover only core arithmetic for complex inputs or fused attention.".into(), "incomplete_operator_formula");
                }
            }
            Err(Error::Uncounted) => gap(
                &name,
                format!("{name} was observed 1 time(s), but no FLOP formula is registered."),
                "uncounted_operator",
            ),
            Err(error) => gap(&name, error.to_string(), "unsupported_operator_formula"),
        }
    }
    let total: BigInt = counts.values().sum();
    let partial = !diagnostics.is_empty();
    let counts: Map<String, Value> = counts.into_iter().map(|(k, v)| (k, number(v))).collect();
    let modules: Map<String, Value> = modules.into_iter().map(|(k, v)| (k, number(v))).collect();
    Ok(
        json!({"schema_version":1,"context":{"torch_version":capture["context"]["torch_version"],"method":METHOD,"counting_convention":"torchscan_flops_v1","boundary":"dispatcher_operator"},
        "total":{"status":if partial {"partial"} else {"complete"},"value":if partial {Value::Null} else {number(total.clone())},"known_value":number(total),"unit":"FLOPs","scope":SCOPE,"method":METHOD},
        "by_operator":counts,"by_module":modules,"ignored_operators":skipped,"diagnostics":diagnostics}),
    )
}
