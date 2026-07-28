import codecs
import os
from pathlib import Path


def root() -> Path:
    return Path(__file__).parent.parent


# file utils
def read(path: Path):
    with codecs.open(os.path.join(path), "r") as fp:
        return fp.read()


def get_version(path: Path = root() / "__about__.py"):
    with path.open("r") as fp:
        for line in fp.readlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
