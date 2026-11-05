// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.

use super::*;

fn dim(shape: &Value, index: usize) -> Result<BigInt> {
    natural(
        array(shape)?
            .get(index)
            .ok_or_else(|| unsupported("Missing shape dimension."))?,
    )
}
fn gcd(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        (a, b) = (b.clone(), a % b);
    }
    a
}
fn convolution(input: &Value, weight: &Value, output: &Value, transposed: bool) -> Result<BigInt> {
    let spatial = array(if transposed { input } else { output })?
        .get(2..)
        .ok_or_else(|| unsupported("Invalid convolution rank."))?;
    Ok(2 * dim(input, 0)? * product(weight)? * product_slice(spatial)?)
}
fn attention_shapes<'a>(
    q: &'a Value,
    k: &'a Value,
    v: &'a Value,
) -> Result<(&'a [Value], &'a [Value], &'a [Value])> {
    let (q, k, v) = (array(q)?, array(k)?, array(v)?);
    if q.len() != 4
        || k.len() != 4
        || v.len() != 4
        || q[0] != k[0]
        || k[..3] != v[..3]
        || q[3] != k[3]
        || natural(&k[1])?.is_zero()
        || !(natural(&q[1])? % natural(&k[1])?).is_zero()
    {
        return Err(unsupported("Attention requires matching batches, divisible query/KV heads, Q/K widths, and K/V lengths."));
    }
    Ok((q, k, v))
}
fn mha(
    q: &Value,
    k: &Value,
    v: &Value,
    width: &Value,
    heads: &Value,
    masked: bool,
    averaged: bool,
) -> Result<BigInt> {
    let shape = array(q)?;
    if shape.len() != 3 || q != k || k != v || shape[2] != *width {
        return Err(unsupported(
            "Fused MHA counts require dense batched self-attention with matching widths.",
        ));
    }
    let (batch, length, width, heads) = (
        natural(&shape[0])?,
        natural(&shape[1])?,
        natural(width)?,
        natural(heads)?,
    );
    let rows = &batch * heads * &length;
    let scores = &rows * &length;
    Ok(8 * &batch * &length * &width * &width
        + &batch * &length * &width
        + 4 * &batch * &length * &length * &width
        + 5 * &scores
        - 2 * rows
        + scores * (u8::from(masked) + u8::from(averaged)))
}

pub(super) fn evaluate(
    kind: &str,
    params: &Map<String, Value>,
    args: &[Value],
    kw: &Map<String, Value>,
    out: &Value,
) -> Result<BigInt> {
    let a = |index| args.get(index).unwrap_or(&Value::Null);
    let opt = |index, name| arg(args, kw, index, name);
    let present = |index, name| u8::from(!opt(index, name).is_null());
    let cost = || scalar(params.get("cost").unwrap_or(&Value::Null), 1.0).map(|n| n as u32);
    match kind {
        "_elementwise" => Ok(cost()? * product(out)?),
        "_activation" => Ok(cost()? * product(a(0))?),
        "_gelu" => {
            let approximation = kw
                .get("approximate")
                .and_then(Value::as_str)
                .unwrap_or("none");
            let cost = match approximation {
                "none" => 5,
                "tanh" => 14,
                _ => {
                    return Err(unsupported(
                        "GELU estimates support only 'none' and 'tanh' approximations.",
                    ))
                }
            };
            Ok(cost * product(out)?)
        }
        "_weight_norm" => {
            let input = array(a(0))?;
            let axis = if opt(2, "dim").is_null() {
                0
            } else {
                integer(opt(2, "dim"))?
                    .to_i64()
                    .ok_or_else(|| unsupported("Invalid dimension."))?
            };
            let rows = if axis == -1 {
                BigInt::one()
            } else if axis >= 0 && (axis as usize) < input.len() {
                natural(&input[axis as usize])?
            } else {
                return Err(unsupported(
                    "WeightNorm FLOPs require a valid dimension or -1 for the whole tensor.",
                ));
            };
            let n = product(a(0))?;
            if product(a(1))? != rows || (!rows.is_zero() && n.is_zero()) {
                return Err(unsupported(
                    "WeightNorm FLOPs require one scale per nonempty norm row.",
                ));
            }
            Ok(3 * n + rows)
        }
        "_pool" => {
            let maximum = params["maximum"] == true;
            let output = if maximum && params.get("indices") != Some(&Value::Bool(false)) {
                a_output(out, 0)?
            } else {
                out
            };
            let kernel = opt(1, "kernel_size");
            let rank = params["rank"].as_u64().expect("Embedded pool rank") as u32;
            let volume = if kernel.is_number() {
                natural(kernel)?.pow(rank)
            } else {
                let kernel = array(kernel)?;
                if kernel.len() == 1 {
                    natural(&kernel[0])?.pow(rank)
                } else {
                    product_slice(kernel)?
                }
            };
            Ok(product(output)? * (volume - u8::from(maximum)))
        }
        "_adaptive_pool" => {
            let maximum = params["maximum"] == true;
            let output = if maximum { a_output(out, 0)? } else { out };
            let (input, output_shape, rank) = (
                array(a(0))?,
                array(output)?,
                array(opt(1, "output_size"))?.len(),
            );
            if rank == 0 || rank > input.len() || rank > output_shape.len() {
                return Err(unsupported("Invalid adaptive pooling rank."));
            }
            let mut visits = product_slice(&input[..input.len() - rank])?;
            for (length, size) in input[input.len() - rank..]
                .iter()
                .zip(&output_shape[output_shape.len() - rank..])
            {
                let (length, size) = (natural(length)?, natural(size)?);
                visits *= &length + &size - gcd(length, size);
            }
            Ok(visits - product(output)? * u8::from(maximum))
        }
        "_rms_norm" => {
            let (input, normalized) = (array(a(0))?, array(opt(1, "normalized_shape"))?);
            if normalized.is_empty()
                || normalized.len() > input.len()
                || input[input.len() - normalized.len()..] != *normalized
                || normalized
                    .iter()
                    .any(|v| natural(v).map_or(true, |n| n.is_zero()))
            {
                return Err(unsupported(
                    "RMSNorm estimates require matching nonempty normalized rows.",
                ));
            }
            if !opt(2, "weight").is_null() && opt(2, "weight") != opt(1, "normalized_shape") {
                return Err(unsupported(
                    "RMSNorm FLOPs require a matching weight shape.",
                ));
            }
            let n = product(a(0))?;
            let rows = &n / product(opt(1, "normalized_shape"))?;
            Ok((3 + present(2, "weight")) * n + 2 * rows)
        }
        "_add" => Ok(product(out)? * (1 + u8::from(scalar(opt(2, "alpha"), 1.0)? != 1.0))),
        "_add_product" => Ok(product(out)? * (2 + u8::from(scalar(opt(3, "value"), 1.0)? != 1.0))),
        "_clamp" => Ok(product(out)? * (present(1, "min") + present(2, "max"))),
        "_elu" => Ok(product(a(0))?
            * (6 + u8::from(scalar(opt(2, "scale"), 1.0)? != 1.0)
                + u8::from(scalar(opt(3, "input_scale"), 1.0)? != 1.0))),
        "_rrelu" => {
            if boolean(opt(4, "training"), false)? {
                return Err(unsupported("RReLU FLOPs cover evaluation calls only."));
            }
            Ok(4 * product(a(0))?)
        }
        "_softmax" | "_log_softmax" | "_safe_softmax" | "_cumsum" => {
            let shape = array(a(0))?;
            let n = product(a(0))?;
            let width = if shape.is_empty() {
                BigInt::one()
            } else {
                natural(&shape[axis(shape, opt(1, "dim"), -1)?])?
            };
            if n.is_zero() || width.is_zero() {
                return Ok(BigInt::zero());
            }
            let rows = &n / width;
            Ok(match kind {
                "_cumsum" => n - rows,
                "_log_softmax" => 4 * n - rows,
                "_safe_softmax" => 5 * n - 2 * rows + 2 * product(out)?,
                _ => 5 * n - 2 * rows,
            })
        }
        "_sum" | "_mean" => {
            let (n, outputs) = (product(a(0))?, product(out)?);
            if kind == "_mean" && n.is_zero() && !outputs.is_zero() {
                return Err(unsupported("Mean FLOPs require nonempty reduction rows."));
            }
            let sum = (n - &outputs).max(BigInt::zero());
            Ok(if kind == "_mean" { sum + outputs } else { sum })
        }
        "_layer_norm" => {
            let (n, width) = (product(a(0))?, product(opt(1, "normalized_shape"))?);
            if width.is_zero() {
                return Err(unsupported(
                    "LayerNorm FLOPs require a nonempty normalized row.",
                ));
            }
            Ok(6 * &n + 2 * (&n / width) + n * (present(2, "weight") + present(3, "bias")))
        }
        "_group_norm" => {
            if product_slice(
                array(a(0))?
                    .get(1..)
                    .ok_or_else(|| unsupported("Invalid input rank."))?,
            )?
            .is_zero()
            {
                return Err(unsupported(
                    "GroupNorm FLOPs require a nonempty normalized group.",
                ));
            }
            let n = product(a(0))?;
            Ok(6 * &n
                + 2 * natural(opt(3, "N"))? * natural(opt(6, "group"))?
                + n * (present(1, "weight") + present(2, "bias")))
        }
        "_batch_norm" | "_batch_norm_legit" | "_batch_norm_eval" => {
            let (n, channels) = (product(a(0))?, dim(a(0), 1)?);
            if n.is_zero() {
                return Err(unsupported("Empty native BatchNorm work depends on strides unavailable to the shape formula."));
            }
            let no_stats = kind == "_batch_norm_legit" && args.len() == 6;
            let training = kind != "_batch_norm_eval"
                && boolean(opt(if no_stats { 3 } else { 5 }, "training"), false)?;
            let mut count = (2 + present(1, "weight") + present(2, "bias")) * &n + 2 * &channels;
            if training {
                count += 4 * n
                    + channels
                        * if no_stats {
                            0
                        } else {
                            3 * present(3, "running_mean") + 5 * present(4, "running_var")
                        };
            }
            Ok(count)
        }
        "_cpu_attention" | "_attention" => {
            let backward = params.get("backward") == Some(&Value::Bool(true));
            let start = usize::from(backward);
            let (q, k, v) = attention_shapes(a(start), a(start + 1), a(start + 2))?;
            let rows = product_slice(&q[..3])?;
            let scores = &rows * natural(&k[2])?;
            let (qwidth, vwidth) = (natural(&q[3])?, natural(&v[3])?);
            if kind == "_attention" {
                return Ok(2
                    * scores
                    * if backward {
                        3 * qwidth + 2 * vwidth
                    } else {
                        qwidth + vwidth
                    });
            }
            if scalar(opt(3, "dropout_p"), 0.0)? != 0.0 {
                return Err(unsupported("CPU attention formula does not cover dropout."));
            }
            if scores.is_zero() {
                return Ok(BigInt::zero());
            }
            Ok(2 * &scores * (qwidth + vwidth) + 6 * &scores - 2 * rows
                + scores
                    * (u8::from(boolean(opt(4, "is_causal"), false)?)
                        + u8::from(!kw.get("attn_mask").unwrap_or(&Value::Null).is_null())))
        }
        "_native_mha" => mha(
            a(0),
            a(1),
            a(2),
            opt(3, "embed_dim"),
            opt(4, "num_head"),
            !opt(9, "mask").is_null(),
            boolean(opt(10, "need_weights"), true)?
                && boolean(opt(11, "average_attn_weights"), true)?,
        ),
        "_native_encoder" => {
            let attention = mha(
                a(0),
                a(0),
                a(0),
                a(1),
                a(2),
                !opt(18, "mask").is_null(),
                false,
            )?;
            let input = array(a(0))?;
            let rows = product_slice(&input[..input.len() - 1])?;
            let elements = product(a(0))?;
            let hidden = &rows * dim(a(14), 0)?;
            Ok(attention
                + 4 * &hidden * natural(a(1))?
                + hidden * if boolean(a(7), false)? { 5 } else { 1 }
                + 18 * elements
                + 4 * rows)
        }
        "_extremum" => {
            if a(1).is_array() {
                return product(out);
            }
            let output = if out
                .as_array()
                .is_some_and(|s| s.first().is_some_and(Value::is_array))
            {
                a_output(out, 0)?
            } else {
                out
            };
            Ok((product(a(0))? - product(output)?).max(BigInt::zero()))
        }
        "_norm" => {
            let order = scalar(
                kw.get("ord")
                    .or_else(|| kw.get("p"))
                    .unwrap_or(opt(1, "order")),
                2.0,
            )?;
            let n = product(a(0))?;
            if order == 2.0 {
                Ok(2 * n)
            } else if order == 1.0 || order.is_infinite() {
                Ok((2_i32 * n - product(out)?).max(BigInt::zero()))
            } else {
                Err(unsupported(
                    "Norm FLOPs cover only orders 1, 2, and +/-infinity.",
                ))
            }
        }
        "_dot" => Ok(2 * product(a(0))?),
        "_grouped_mm" => {
            let (input, weight) = (array(a(0))?, array(a(1))?);
            if input.len() != 2 || weight.len() != 3 || input[1] != weight[1] {
                return Err(unsupported(
                    "Grouped matmul counts cover packed rows and equal-width expert matrices only.",
                ));
            }
            Ok(2 * product(a(0))? * natural(&weight[2])?)
        }
        "_product" => {
            let left = params["left"].as_u64().expect("Embedded product index") as usize;
            Ok(2 * product(out)?
                * natural(
                    array(a(left))?
                        .last()
                        .ok_or_else(|| unsupported("Missing matrix dimensions."))?,
                )?)
        }
        "_convolution" => convolution(
            a(0),
            a(1),
            out,
            if let Some(index) = params.get("transposed_arg").and_then(Value::as_u64) {
                boolean(a(index as usize), false)?
            } else {
                false
            },
        ),
        "_convolution_backward" => {
            let mask = array(a(10))?;
            let requested = mask
                .first()
                .and_then(Value::as_bool)
                .ok_or_else(|| unsupported("Missing convolution gradient mask."))?;
            let weight = mask
                .get(1)
                .and_then(Value::as_bool)
                .ok_or_else(|| unsupported("Missing convolution gradient mask."))?;
            Ok(convolution(a(1), a(2), a(0), boolean(a(7), false)?)?
                * (u8::from(requested) + u8::from(weight)))
        }
        _ => Err(Error::Uncounted),
    }
}
fn a_output(output: &Value, index: usize) -> Result<&Value> {
    array(output)?
        .get(index)
        .ok_or_else(|| unsupported("Missing operator output shape."))
}
