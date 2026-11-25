"""Execute the intro notebook and write an executed copy plus figure artifacts."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as error:
        raise SystemExit("Install the notebook extras first: python -m pip install -e '.[notebook]'") from error
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "outputs/intro/intro.executed.ipynb"
    notebook = nbformat.read(root / "examples/intro.ipynb", as_version=4)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}})
    client.execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Executed notebook: {output}")
    print(f"Figures and metrics: {root / 'outputs/intro'}")


if __name__ == "__main__":
    main()
