import unittest
from unittest.mock import patch

from lcip_app.gui_support import GuiDependencyError, build_gui_import_error


class GuiSupportTests(unittest.TestCase):
    @patch("lcip_app.gui_support._detect_linux_distribution", return_value="arch")
    def test_libxkbcommon_error_includes_arch_fix(self, _detect_distro):
        error = build_gui_import_error(
            ImportError("libxkbcommon.so.0: cannot open shared object file: No such file or directory")
        )

        self.assertIsInstance(error, GuiDependencyError)
        self.assertIn("libxkbcommon.so.0", str(error))
        self.assertIn("sudo pacman -S libxkbcommon", str(error))

    def test_generic_gui_import_error_is_preserved(self):
        error = build_gui_import_error(ImportError("some other import failure"))
        self.assertIsInstance(error, GuiDependencyError)
        self.assertIn("some other import failure", str(error))


if __name__ == "__main__":
    unittest.main()
