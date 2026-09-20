//! JSON transport shared by the Python frontend and command-line callers.
use serde_json::json;
use std::io::{self, Read};
use xir_core::{deserialize_module, verify_module, Diagnostic, DiagnosticCode, ModuleBuilder};

fn run() -> Result<serde_json::Value, String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 || !matches!(args[0].as_str(), "construct" | "verify" | "emit-cuda") {
        return Err("usage: xir-verify construct|verify|emit-cuda < module.json".into());
    }
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| error.to_string())?;
    let module = match deserialize_module(&input) {
        Ok(module) => module,
        Err(error) => {
            return Ok(
                json!({"ok": false, "diagnostics": [Diagnostic::error(DiagnosticCode::ParseError, error.to_string())]}),
            )
        }
    };
    if args[0] == "construct" {
        Ok(match ModuleBuilder::from_module(module).finish_checked() {
            Ok(module) => json!({"ok": true, "module": module, "diagnostics": []}),
            Err(errors) => json!({"ok": false, "diagnostics": errors.diagnostics}),
        })
    } else if args[0] == "emit-cuda" {
        Ok(match xir_core::cuda::emit_cuda(&module) {
            Ok(artifact) => json!({"ok": true, "artifact": artifact}),
            Err(errors) => json!({"ok": false, "diagnostics": errors.diagnostics}),
        })
    } else {
        let report = verify_module(&module);
        Ok(
            json!({"ok": !matches!(report.construction, xir_core::CheckStatus::Fail), "report": report}),
        )
    }
}

fn main() {
    match run() {
        Ok(response) => println!("{}", response),
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    }
}
