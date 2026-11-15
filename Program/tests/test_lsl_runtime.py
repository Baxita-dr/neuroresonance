from src.lsl_runtime import classify_lsl_import_error


def test_lsl_diagnostic_distinguishes_missing_python_package_from_native_library():
    package_kind, package_message = classify_lsl_import_error(
        ModuleNotFoundError("No module named 'pylsl'", name="pylsl")
    )
    native_kind, native_message = classify_lsl_import_error(OSError("liblsl.dll not found"))

    assert package_kind == "package_missing"
    assert "Python package 'pylsl' is not installed" in package_message
    assert native_kind == "native_library_unavailable"
    assert "pylsl' is installed" in native_message
    assert "liblsl.dll" in native_message
