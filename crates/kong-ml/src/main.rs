use safetensors::SafeTensors;
use std::io::Error;
fn main() -> Result<(), Error> {
    let paths = vec![
        "../../tools/xda/models/xda_x86_64.safetensors",
        "../../tools/xda/models/xda_aarch64.safetensors",
        "../../tools/xda/models/xda_arm.safetensors",
        "../../tools/xda/models/xda_riscv64.safetensors",
    ];
    for path in paths {
        let print_path = path.split("/").last().unwrap();
        println!("==========={print_path}===========");
        let buffer = &std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(buffer).unwrap();
        for name in tensors.names() {
            println!("{name}");
        }
    }
    Ok(())
}
