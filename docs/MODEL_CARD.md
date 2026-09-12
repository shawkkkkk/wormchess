# WormChess model card

## Intended claim

A legal chess move is selected by a deterministic policy whose candidate
states propagate over the complete Cook 2019 connectome topology.

## Claims not made

The worm understands chess; the dynamics reproduce living _C. elegans_ neural
activity; the policy is trained; the project demonstrates intelligence or
consciousness; or the connectome beats ordinary baselines.

## Evaluation gate

Before labeling a future checkpoint “trained,” freeze it and publish games on
unseen seeds against random, feature-only, no-edge, and relabeled-graph
controls. Report legal-move rate, win/draw/loss, average material, game length,
all hyperparameters, and failed runs.
