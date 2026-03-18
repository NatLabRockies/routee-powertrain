# Build The Docs

The documentation is built using jupyter-book which you can install with pip:

```bash
pip install jupyter-book
```

Make sure to convert the examples to jupyter notebooks:

```bash
python docs/examples/_convert_examples_to_notebooks.py
```

Then, to build the docs, run the following command from the root of the repository:

```bash
jupyter-book build docs/
```

This, build build the docs as html and place them in `docs/_build/html/`. You can then view the docs by opening `docs/_build/html/index.html` in your browser.
