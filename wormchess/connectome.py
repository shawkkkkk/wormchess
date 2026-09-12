from pathlib import Path

import numpy as np


GRAPH = Path(__file__).resolve().parents[1] / "data" / "connectome.npz"


class Connectome:
    def __init__(self, path=GRAPH):
        with np.load(path, allow_pickle=False) as data:
            names = np.asarray(data["neurons"]).astype(str)
            chemical = np.asarray(data["chemical"], np.float32)
            electrical = np.asarray(data["electrical"], np.float32)
        if len(names) != 302:
            raise ValueError("WormChess requires the complete 302-node graph.")
        incoming = chemical.T + electrical.T
        totals = incoming.sum(axis=1, keepdims=True)
        matrix = np.divide(incoming, totals, out=np.zeros_like(incoming), where=totals > 0)
        matrix.setflags(write=False)
        self.names = tuple(names.tolist())
        self.lookup = {name: i for i, name in enumerate(self.names)}
        self.matrix = matrix

    def index(self, name):
        return self.lookup[name]
