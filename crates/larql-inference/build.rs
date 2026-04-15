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
}
