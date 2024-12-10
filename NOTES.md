# Notes

## Auto reload for local imports in Jupyter Notebooks
Works generally for all imports.
```python
%load_ext autoreload
%autoreload 2
```

## Using MiniZinc in Jupyter Notebooks
```python
import nest_asyncio
nest_asyncio.apply()
```