fn main() {
    // Windows: link OpenBLAS statically from vcpkg x64-windows-static.
    // openblas-src is not used on Windows; we emit the link directives here instead.
    #[cfg(windows)]
    {
        let vcpkg_root = std::env::var("VCPKG_ROOT")
            .expect("VCPKG_ROOT must be set to your vcpkg installation directory");
        let lib_dir = format!("{}/installed/x64-windows-static/lib", vcpkg_root.replace('\\', "/"));
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:rustc-link-lib=static=openblas");
        println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    }
    // Rebuild if anything under csrc/ changes (new .c, new .h, modified source).
    // The cc crate only auto-tracks files passed to .file(); this widens the net so
    // a new or modified C source always triggers recompilation of q4_dot.
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");

    let mut build = cc::Build::new();
    build.file("csrc/q4_dot.c");
    build.opt_level(3);

    #[cfg(target_arch = "aarch64")]
    build.flag("-march=armv8.2-a+dotprod");

    #[cfg(target_arch = "x86_64")]
    {
        // MSVC uses /arch:AVX2; GCC/Clang use -mavx2
        if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
            build.flag("/arch:AVX2");
        } else {
            build.flag("-mavx2");
        }
    }

    build.compile("q4_dot");
}
