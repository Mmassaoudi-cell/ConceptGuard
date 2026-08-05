# ConceptGuard-RL

Source code for **ConceptGuard-RL**, a concept-bottleneck reinforcement-learning policy for
network intrusion response. The policy operates over a partially-observable five-action
environment (classify benign, classify attack, abstain, request telemetry at a cost, escalate to
an analyst) and routes its state through predicted, human-meaningful concepts rather than raw
packet features, with a working intervention mechanism for analyst correction.

This repository also includes the source code for six other candidate RL/graph architectures
screened for the same task (CrossGuard-RL, TTA-GraphRL, TailGuard, HyperGuard-RL, RiemannGuard,
CDE-Guard) and the baseline/statistical-testing scripts they were compared against, for full
transparency into the selection process — not just the final method.

This repository contains source code only: no datasets, figures, tables, results, or manuscript
files are included.

## Repository structure

```
project/code/
  common/                     Shared infrastructure
    data_utils.py               Leakage-safe loading/splitting/preprocessing
    dataset_configs.py          Per-dataset config (paths, label columns, id-like columns)
    device_shift_loader.py      Real benign-distribution-shift split (Edge-IIoTset)
    graph_env.py                Shared 5-action POMDP environment + graph encoders (SAGE/GCN/GAT)
    dqn.py                      DQN trainer
    qr_dqn.py                   Quantile-Regression DQN (distributional RL) trainer
    encoders.py                 Graph encoder pretraining + Tent test-time adapter
    hypergraph.py                Typed hypergraph encoder
    hyperbolic.py                Poincare-ball hyperbolic encoder
    multimodal.py                Modality-block fusion encoder
    timedecay.py                  Time-decay GRU (continuous-time-aware recurrent encoder)

  candidates/
    conceptguard_rl.py           ConceptGuard-RL (the proposed/selected method)
    crossguard_rl.py             CrossGuard-RL (missing-modality-robust contrastive fusion)
    tta_graphrl.py                TTA-GraphRL (test-time-adaptive graph RL)
    tta_graphrl_deviceshift.py    TTA-GraphRL device-holdout shift experiment
    tailguard.py                  TailGuard (distributional/CVaR risk-sensitive RL)
    hyperguard_rl.py              HyperGuard-RL (temporal hypergraph RL)
    riemannguard.py                RiemannGuard (hyperbolic curvature RL)
    cdeguard.py                    CDE-Guard (time-decay graph RL)

  baselines/
    run_classical_baselines.py     Initial 4-model classical sweep
    run_classical_baselines_v2.py  Extended 14-model classical/tree/MLP sweep
    run_deep_baselines.py          Deep sequence baselines (1D-CNN, LSTM, GRU, Transformer)
    run_rl_graph_baselines.py      Graph (GCN/GraphSAGE/GAT) and RL (random policy, vanilla DQN) baselines
    statistical_tests.py            Paired Wilcoxon + Holm correction significance testing

  screen_datasets.py              Dataset screening utility
  make_figures.py                 Plotting code for result figures (not the figures themselves)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires a local dataset collection (UNSW-NB15, Edge-IIoTset, Bot_IoT, CICIoT23, RT-IoT2022, etc.)
laid out per the paths in `project/code/common/dataset_configs.py`.

## Known limitation: hardcoded local paths

This code was developed and run on a single research workstation, and several files contain
hardcoded absolute Windows paths (dataset locations in `dataset_configs.py`; results/log output
directories in the `candidates/` and `baselines/` scripts). **These must be edited to match your
own environment before running.** This is a known reproducibility gap, not an oversight —
parameterizing them via environment variables or a config file is a natural follow-up.

## Usage

Each candidate script is runnable standalone, e.g.:

```bash
python project/code/candidates/conceptguard_rl.py --dataset unsw_nb15 --seed 42 --device cuda
```

See each script's `argparse` block for its specific arguments (seed, episode/window counts,
device, dataset-specific options).

## License

MIT — see `LICENSE`.
