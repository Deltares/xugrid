import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import pytest


def on_ci():
    """
    Don't test the examples on CI: they run during building of documentation.
    """
    return os.environ.get("GITHUB_ACTION") is not None


def get_examples():
    # Where are we? --> __file__
    # Move two up.
    path = Path(__file__).parent.parent
    relpath = Path(os.path.relpath(path, os.getcwd())) / "examples/*.py"
    examples = [f for f in glob(str(relpath)) if f.endswith(".py")]
    return examples


@pytest.mark.parametrize("example", get_examples())
@pytest.mark.skipif(on_ci(), reason="Examples are run during docs build")
def test_example(example):
    subprocess.run([sys.executable, example], check=True)
