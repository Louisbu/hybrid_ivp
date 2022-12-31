"""
Generate all the figures used in the paper
"""

from pathlib import Path

import typer


def main(path_out: str = "output"):
    # Build output folder
    path_out: Path = Path(path_out)
    if not path_out.exists():
        path_out.mkdir()


if __name__ == "__main__":
    typer.run(main)
