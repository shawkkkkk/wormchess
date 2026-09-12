"""Generate the compact browser adjacency file from the audited NPZ graph."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with np.load(ROOT / "data" / "connectome.npz", allow_pickle=False) as data:
        names = np.asarray(data["neurons"]).astype(str)
        incoming = np.asarray(data["chemical"], np.float32).T + np.asarray(data["electrical"], np.float32).T
    totals = incoming.sum(axis=1, keepdims=True)
    matrix = np.divide(incoming, totals, out=np.zeros_like(incoming), where=totals > 0)
    rows = []
    for row in matrix:
        rows.append([[int(i), round(float(row[i]), 7)] for i in np.flatnonzero(row)])
    payload = {
        "source": "Cook et al. 2019 hermaphrodite via OpenWorm cect",
        "orientation": "incoming[post] = [pre, normalized_weight]",
        "neurons": names.tolist(),
        "incoming": rows,
        "interface": {
            "seed": 302,
            "sensory": ["ASEL", "ASER", "AWAL", "AWAR", "AWCL", "AWCR", "ASHL", "ASHR", "AFDL", "AFDR"],
            "projection": np.random.default_rng(302).normal(
                0, 1 / np.sqrt(10), (10, 10)
            ).round(7).tolist(),
            "readout": ["AVAL", "AVAR", "AVBL", "AVBR", "PVCL", "PVCR", "RIVL", "RIVR"],
            "weights": [-0.9, -0.8, 1.0, 0.9, 0.65, 0.7, -0.2, 0.2],
        },
    }
    target = ROOT / "site" / "connectome.json"
    target.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {target} ({len(names)} neurons)")


if __name__ == "__main__":
    main()
