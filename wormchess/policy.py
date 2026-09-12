from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome


SENSORY = ("ASEL", "ASER", "AWAL", "AWAR", "AWCL", "AWCR", "ASHL", "ASHR", "AFDL", "AFDR")
READOUT = ("AVAL", "AVAR", "AVBL", "AVBR", "PVCL", "PVCR", "RIVL", "RIVR")


@dataclass(frozen=True)
class MoveFeatures:
    from_file: float
    from_rank: float
    to_file: float
    to_rank: float
    piece: float
    capture: float = 0.0
    promotion: float = 0.0
    check: float = 0.0
    center: float = 0.0
    development: float = 0.0

    def array(self) -> np.ndarray:
        values = np.asarray(tuple(self.__dict__.values()), dtype=np.float32)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError("Move features must contain ten finite values.")
        return values


class WormChessPolicy:
    """Score legal candidates. It does not generate or validate chess moves."""

    def __init__(self, graph: Connectome | None = None, seed: int = 302):
        self.graph = graph or Connectome()
        self.seed = seed
        rng = np.random.default_rng(seed)
        projection = rng.normal(0, 1 / np.sqrt(10), (len(SENSORY), 10)).astype(np.float32)
        projection.setflags(write=False)
        self.projection = projection
        self.state = np.zeros(302, np.float32)
        # Declared fixed interface weights, not learned chess knowledge.
        self.readout = np.asarray([-0.9, -0.8, 1.0, .9, .65, .7, -.2, .2], np.float32)

    def reset(self) -> None:
        self.state.fill(0)

    def evaluate(self, features: MoveFeatures, no_edges: bool = False) -> tuple[float, np.ndarray]:
        drive = np.zeros(302, np.float32)
        projected = np.tanh(self.projection @ features.array())
        for name, value in zip(SENSORY, projected):
            drive[self.graph.index(name)] = value
        candidate = self.state.copy()
        for _ in range(6):
            recurrence = np.zeros_like(candidate) if no_edges else self.graph.matrix @ candidate
            candidate = np.tanh(.72 * recurrence + .58 * drive).astype(np.float32)
        read = np.asarray([candidate[self.graph.index(name)] for name in READOUT])
        # Tiny explicit chess feature terms prevent completely degenerate play;
        # they are visible plumbing, not a hidden search engine.
        score = float(read @ self.readout + .08 * features.capture + .05 * features.center + .12 * features.promotion)
        return score, candidate

    def choose(self, candidates: list[tuple[str, MoveFeatures]], no_edges: bool = False) -> tuple[str, list[dict]]:
        if not candidates:
            raise ValueError("The rules engine supplied no legal candidates.")
        scored = []
        states = []
        for move, features in candidates:
            score, state = self.evaluate(features, no_edges=no_edges)
            scored.append({"move": move, "score": score})
            states.append(state)
        order = sorted(range(len(scored)), key=lambda i: (-scored[i]["score"], scored[i]["move"]))
        winner = order[0]
        self.state = states[winner]
        return scored[winner]["move"], [scored[i] for i in order]
