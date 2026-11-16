"""PyInstaller entry point (absolute imports; __main__.py uses relative imports).

One frozen binary, two jobs:
    aieditor-engine.exe [--port N]          desktop engine (what the app launches)
    aieditor-engine.exe serve | reel | ...  the `aieditor` command line (aieditor.cmd forwards here)
"""

import multiprocessing
import sys
from pathlib import Path

CLI_COMMANDS = {"serve", "run", "reel", "highlights", "captions", "edit", "photo", "render", "transcribe", "watch",
                "keys", "recipes", "styles", "mcp", "licence", "license", "--help", "-h", "--home", "--log-level"}


def main() -> None:
    multiprocessing.freeze_support()
    if Path(sys.argv[0]).stem.lower() == "aieditor" or (len(sys.argv) > 1 and sys.argv[1] in CLI_COMMANDS):
        from aieditor_engine.cli import main as cli

        raise SystemExit(cli(sys.argv[1:]))
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from aieditor_engine.cli import main as cli

        raise SystemExit(cli(sys.argv[2:]))
    from aieditor_engine.__main__ import main as engine

    engine()


if __name__ == "__main__":
    main()
