from unittest.mock import patch

from lcip_app.gui_support import GuiDependencyError, build_gui_import_error


@patch("lcip_app.gui_support._detect_linux_distribution", return_value="arch")
def test_libxkbcommon_error_includes_arch_fix(_detect_distro):
    error = build_gui_import_error(
        ImportError("libxkbcommon.so.0: cannot open shared object file: No such file or directory")
    )

    assert isinstance(error, GuiDependencyError)
    assert "libxkbcommon.so.0" in str(error)
    assert "sudo pacman -S libxkbcommon" in str(error)


def test_generic_gui_import_error_is_preserved():
    error = build_gui_import_error(ImportError("some other import failure"))
    assert isinstance(error, GuiDependencyError)
    assert "some other import failure" in str(error)
