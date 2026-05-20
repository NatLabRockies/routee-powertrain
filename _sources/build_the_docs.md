# Build The Docs

The documentation is built using jupyter-book V1 which you can install with pip:

```bash
pip install jupyter-book<2
```

Then, to build the docs, run the following command from the root of the repository:

```bash
jupyter-book build docs/
```

This builds the docs as `.html` and places them in `docs/_build/html/`. You can then view the docs by opening `docs/_build/html/index.html` in your browser.