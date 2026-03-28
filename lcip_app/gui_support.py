from pathlib import Path


class GuiDependencyError(RuntimeError):
    pass


def _detect_linux_distribution() -> str | None:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return None

    values = {}
    for line in os_release.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values.get("ID")


def build_gui_import_error(exc: ImportError) -> RuntimeError:
    error_text = str(exc)
    if "libxkbcommon.so.0" not in error_text:
        return GuiDependencyError(f"Failed to import the Qt GUI dependencies: {error_text}")

    distro_id = _detect_linux_distribution()
    install_hint = "Install the system package that provides libxkbcommon.so.0."
    if distro_id == "arch":
        install_hint = "Install the missing system package with `sudo pacman -S libxkbcommon`."
    elif distro_id in {"ubuntu", "debian"}:
        install_hint = "Install the missing system package with `sudo apt install libxkbcommon0`."
    elif distro_id in {"fedora", "rhel", "centos"}:
        install_hint = "Install the missing system package with `sudo dnf install libxkbcommon`."

    return GuiDependencyError(
        "PySide6 is installed, but Qt could not load a required system library: "
        "`libxkbcommon.so.0`. "
        f"{install_hint}"
    )


def import_basic_gui():
    try:
        from gui import LCIP_GUI_Basic
    except ImportError as exc:
        raise build_gui_import_error(exc) from exc
    return LCIP_GUI_Basic


def import_gan_gui():
    try:
        from gui import LCIP_GUI_GAN
    except ImportError as exc:
        raise build_gui_import_error(exc) from exc
    return LCIP_GUI_GAN
