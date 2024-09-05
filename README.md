# General Linear Model for Bootcamp Dataset

## How to run

The GLM is run in a single script (`code/glm.py`):

```shell
python code/glm.py
```

The dependencies are listed in a comment at the top. If you have
[pipx](https://pipx.pypa.io/), [hatch](https://hatch.pypa.io/),
[pip-run](https://pip-run.readthedocs.io) or [uv](https://docs.astral.sh/uv/)
installed, you can run with one of the following:

```
pipx run code/glm.py
hatch run code/glm.py
pip run code/glm.py
uv run code/glm.py
```

If you have `uv` installed, `./code/glm.py` will automatically use it.

## Dataset structure

- Raw dataset may be found at `sourcedata/raw`.
- Derivative datasets may be found at `sourcedata/derivatives`.
- GLM code is located in `code/`.
