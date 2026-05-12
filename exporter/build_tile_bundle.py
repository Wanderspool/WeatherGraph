from __future__ import annotations

import sys

from weathergraph.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["build-tile-bundle", *sys.argv[1:]]))