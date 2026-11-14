fn main() {
    if let Err(error) = vir_tui::run() {
        eprintln!("vir-tui: {error}");
        std::process::exit(1);
    }
}
