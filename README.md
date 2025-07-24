intra-class-rare-learn - A scikit-learn compatible intra-class rarity learning package
============================================================

![tests](https://github.com//jannewer/intra-class-rare-learn/actions/workflows/python-app.yml/badge.svg)
![doc](https://github.com//jannewer/intra-class-rare-learn/actions/workflows/deploy-gh-pages.yml/badge.svg)

**intra-class-rare-learn** is a scikit-learn compatible intra-class rarity learning package.
It is based on the [template project](https://github.com/scikit-learn-contrib/project-template)
for [scikit-learn](https://scikit-learn.org) compatible extensions,
but was modified to use [uv](https://docs.astral.sh/uv/) instead of pixi.

## Documentation

Documentation is available at https://jannewer.github.io/intra-class-rare-learn/

## Development
For development, make sure you have uv installed: https://docs.astral.sh/uv/getting-started/installation/

Afterwards, you can do the following:
- run the tests with `uv run task test`
- build the documentation with `uv run task build-doc`
- run black formatting with `uv run task black`
- run ruff linting and formatting with `uv run task ruff`
- run both black and ruff with `uv run task lint`
