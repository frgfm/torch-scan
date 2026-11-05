// Copyright (C) 2020-2026, François-Guillaume Fernandez.
// Licensed under the Apache License 2.0. See LICENSE for details.

use std::{
    env, fs,
    io::{self, Write},
    process::ExitCode,
};

fn count() -> Result<bool, Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("Usage: torchscan-count capture.json [-o report.json]\nCounts saved operator metadata; never runs a model.\nExit codes: 0 complete, 2 partial or invalid input.");
        return Ok(true);
    }
    if args.len() != 1 && !(args.len() == 3 && args[1] == "-o") {
        return Err("Usage: torchscan-count capture.json [-o report.json]".into());
    }
    let capture = serde_json::from_slice(&fs::read(&args[0])?)?;
    let report = torchscan_count::analyze(&capture)?;
    let mut output = serde_json::to_vec_pretty(&report)?;
    output.push(b'\n');
    if args.len() == 3 {
        fs::write(&args[2], output)?;
    } else {
        io::stdout().lock().write_all(&output)?;
    }
    Ok(report["total"]["status"] == "complete")
}
fn main() -> ExitCode {
    match count() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("Incomplete count; see report diagnostics.");
            ExitCode::from(2)
        }
        Err(error) => {
            eprintln!("torchscan-count: {error}");
            ExitCode::from(2)
        }
    }
}
