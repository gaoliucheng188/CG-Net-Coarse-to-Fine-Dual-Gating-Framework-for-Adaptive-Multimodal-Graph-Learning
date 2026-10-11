# NetworkX compatibility shim.
#
# This legacy codebase was written for NetworkX 2.x and calls the top-level
# helpers ``nx.read_gpickle`` / ``nx.write_gpickle``.  These were removed from
# the top-level namespace in NetworkX >= 3.2 and the ``networkx.readwrite.gpickle``
# module was deleted entirely in 3.2.1.  Since gpickle is literally a pickled
# ``nx.Graph``, we restore both helpers on top of ``pickle`` once at import time
# (any ``import utils.*`` triggers this module first), without touching the
# conda environment or the 11+ call sites.
import pickle

import networkx as nx


def _ensure_networkx_compat():
    if not hasattr(nx, "read_gpickle"):
        def _read_gpickle(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        nx.read_gpickle = _read_gpickle
    if not hasattr(nx, "write_gpickle"):
        def _write_gpickle(G, path):
            with open(path, "wb") as f:
                pickle.dump(G, f)
        nx.write_gpickle = _write_gpickle


_ensure_networkx_compat()
