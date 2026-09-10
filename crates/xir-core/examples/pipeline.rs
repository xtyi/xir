mod support;
fn main() {
    println!(
        "{}",
        xir_core::serialize_module(&support::pipeline_module()).unwrap()
    );
}
