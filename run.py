import sys

from lcip_app.cli import main
from lcip_app.gui_support import GuiDependencyError


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GuiDependencyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
