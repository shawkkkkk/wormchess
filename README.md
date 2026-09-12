# WormChess

**Play chess against a fixed 302-neuron _C. elegans_ connectome controller.**

WormChess is a playable, browser-first experiment. A conventional rules engine
generates only legal moves. Every candidate move is converted into sensory
features, passed through the Cook et al. (2019) connectome, and scored by an
explicit neural readout. The selected state becomes the worm's memory for the
next turn.

The interface is inspired by the public FlyChess demonstration—especially the
idle neural silhouette and visible firing while the animal is “thinking”—but
the design and code here are an independent worm-specific implementation.

## Play locally

```bash
python -m http.server 8000 -d site
```

Open `http://localhost:8000`. You play White. Click a piece and a destination;
promotion defaults to a queen. The site loads the bundled 302-node graph and
uses the vendored `chess.js` rules library, so it does not call an AI service or
chess engine.

## What makes the move

1. `chess.js` supplies the legal-move list and game-over rules.
2. Each legal move becomes public features: piece, squares, capture, promotion,
   check, center distance, and development.
3. Those fields stimulate named sensory neurons through a fixed projection.
4. The complete graph advances for six abstract recurrence steps per move.
5. A fixed readout of AVA/AVB/PVC/RIV and pooled activity scores each candidate.
6. The top score is played and its state is retained.

The rules engine **does not** recommend a move. There is no Stockfish, minimax,
opening book, or hidden LLM. Legal-move filtering is ordinary game plumbing,
not worm intelligence.

## Current status

**Phase 1: playable untrained connectome policy.**

- full real graph bundled for Python and exported to the browser;
- browser game with legal moves, restart, move log, thinking animation, and
  idle/active neural silhouette;
- Python policy and training-data boundary;
- deterministic and graph-free control tests;
- no claim that this policy is strong or trained.

The optional training harness can fit only a small readout against declared
examples; it never changes anatomy. A future trained checkpoint must include
held-out games against random, material-only, no-edge, and shuffled-graph
controls before the website can say “trained.”

## Python experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m wormchess.demo
pytest
```

## Scientific boundary

| Component | What it is |
|---|---|
| Cook graph | measured anatomical connectivity |
| move features → sensory neurons | engineered |
| recurrence state | abstract numerical model |
| legal-move generation | conventional chess rule engine |
| move score/readout | engineered, currently untrained |
| visible silhouette | state visualization, not consciousness |

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) and [THIRD_PARTY.md](THIRD_PARTY.md).

MIT licensed.
