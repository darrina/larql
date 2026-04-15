fn main() -> Result<(), Box<dyn std::error::Error>> {
    // On Linux/macOS use the bundled protoc built from source.
    // On Windows, protobuf-src fails to compile (MSVC/Abseil CRT issue), so
    // tonic_build picks up protoc from PATH — install via: winget install Google.Protobuf
    #[cfg(not(windows))]
    std::env::set_var("PROTOC", protobuf_src::protoc());

    tonic_build::compile_protos("proto/vindex.proto")?;
    Ok(())
}
