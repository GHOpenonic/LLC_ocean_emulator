"""Open an emulator predictions.zarr with its per-level fields stacked along k."""
import re

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

_LEVEL = re.compile(r"^(?P<field>.+)_(?P<level>\d+)$")


class _StackedLevels(BackendArray):
    """(time, k, ...) over per-level (time, ...) variables; reads only the levels asked for."""

    def __init__(self, levels):
        self.levels = levels
        self.shape = (levels[0].shape[0], len(levels), *levels[0].shape[1:])
        self.dtype = levels[0].dtype

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem)

    def _getitem(self, key):
        t, k, *rest = key
        picked = range(len(self.levels))[k]
        if isinstance(picked, int):
            return np.asarray(self.levels[picked][(t, *rest)])
        return np.stack([np.asarray(self.levels[i][(t, *rest)]) for i in picked],
                        axis=0 if isinstance(t, int) else 1)


def open_predictions(path, chunks=None):
    """U, V, Theta, Salt, ... as (time, k, [face,] lat, lon) and Eta as (time, [face,] lat, lon).

    Lazily indexed by default; chunks={} gives dask arrays with one level per chunk on the store's tiling.
    """
    flat = xr.open_zarr(path, chunks=None)
    levels = {}
    for name in flat.data_vars:
        m = _LEVEL.match(str(name))
        if m:
            levels.setdefault(m["field"], []).append((int(m["level"]), name))

    first = next(iter(levels.values()))
    dims = flat[first[0][1]].dims  # (time, [face,] lat, lon)
    fields = {}
    for field, pairs in levels.items():
        pairs.sort()
        data = indexing.LazilyIndexedArray(_StackedLevels([flat[n].variable for _, n in pairs]))
        fields[field] = xr.Variable(("time", "k", *dims[1:]), data)
    for name in flat.data_vars:
        if not _LEVEL.match(str(name)):
            fields[name] = flat[name].variable

    ds = xr.Dataset(fields, coords={"k": [lev for lev, _ in first], **flat.coords})
    ds.attrs.update(flat.attrs)
    if chunks is not None:
        ds = ds.chunk({**dict(zip(dims, flat[first[0][1]].encoding["chunks"])), "k": 1, **chunks})
    return ds
