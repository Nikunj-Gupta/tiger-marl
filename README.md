# TIGER-MARL

**T**emporal **I**nformation through **G**raph-based **E**mbeddings and **R**epresentations for Multi-Agent Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2511.08832-b31b1b.svg)](https://arxiv.org/abs/2511.08832)

## Overview

TIGER is a multi-agent reinforcement learning (MARL) framework that enhances cooperative learning by building **dynamic temporal graphs** over agent interactions. Instead of relying on static coordination graphs, TIGER uses a Temporal Graph Attention Network (TGAT) to track how agent relationships evolve over time and produce time-aware agent embeddings that guide value function factorization.

TIGER integrates with two backbone architectures:
- **TIGER-MIX** (`tiger_mix`): Temporal graph attention over a QMIX mixing network
- **TIGER-DICG** (`tiger_dicg`): Temporal graph attention combined with Deep Implicit Coordination Graphs

### How It Works

1. **Static graph construction** — For each batch of episode trajectories, a fully connected graph is built over agents at each timestep.
2. **Attention-based edge filtering** — A GATv2 layer scores edges; the top `(1 - k_percent)` fraction by attention weight is retained, forming a sparse dynamic graph.
3. **Temporal edge generation** — Filtered edges are extended across time using `k_past_neighbors` (cross-agent temporal links) and `self_past` (self-recurrent links), producing a temporal graph.
4. **TGAT aggregation** — A Temporal Graph Attention Network aggregates information over the temporal neighborhood, yielding time-aware agent representations.
5. **QMIX mixing** — The enriched representations feed into the standard QMIX monotonic mixing network to produce the joint value estimate.

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python ≥ 3.8
- PyTorch ≥ 2.0 (with CUDA 12.1 recommended)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) ≥ 2.4.0
- [Sacred](https://sacred.readthedocs.io/) (experiment tracking)
- [PettingZoo](https://pettingzoo.farama.org/) + [Gymnasium](https://gymnasium.farama.org/)

## Usage

Training is launched via `src/main.py` using [Sacred](https://sacred.readthedocs.io/) for configuration management.

### Basic Command

```bash
python src/main.py --config=<alg> --env-config=<env> with [overrides]
```

### TIGER-MIX on Gather

```bash
python src/main.py --config=tiger_mix --env-config=gather \
  with seed=0 use_cuda=True \
       k_percent=0.9 self_past=1 k_past_neighbors=2
```

### TIGER-DICG on Gather

```bash
python src/main.py --config=tiger_dicg --env-config=gather \
  with seed=0 use_cuda=True \
       k_percent=0.9 self_past=1 k_past_neighbors=2
```

### TIGER-MIX on Tag (PettingZoo MPE)

```bash
python src/main.py --config=tiger_mix --env-config=gymma \
  with env_args.key="pz-mpe-simple-tag-v3" seed=0 use_cuda=True \
       k_percent=0.9 self_past=1 k_past_neighbors=2
```

### TIGER-DICG on Tag (PettingZoo MPE)

```bash
python src/main.py --config=tiger_dicg --env-config=gymma \
  with env_args.key="pz-mpe-simple-tag-v3" seed=0 use_cuda=True \
       k_percent=0.9 self_past=1 k_past_neighbors=2
```

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `k_percent` | Attention quantile threshold for edge filtering. Edges with attention weight ≥ this quantile are kept. Higher values = sparser graphs. | `0.9` |
| `self_past` | Number of past timesteps for self-temporal edges (agent connected to its own past representations). | `1` |
| `k_past_neighbors` | Number of past timesteps for cross-agent temporal edges. | `2` |
| `use_cuda` | Whether to use GPU. | `True` |
| `seed` | Random seed. | `0` |

Standard QMIX hyperparameters (`mixing_embed_dim`, `hypernet_layers`, `hypernet_embed`, etc.) are configured in `src/config/algs/tiger_mix.yaml`.

## Repository Structure

```
tiger-marl/
├── src/
│   ├── main.py                        # Entry point (Sacred experiment runner)
│   ├── run.py                         # Training loop
│   ├── config/
│   │   ├── default.yaml               # Global default hyperparameters
│   │   ├── algs/
│   │   │   ├── tiger_mix.yaml         # TIGER-MIX configuration
│   │   │   └── tiger_dicg.yaml        # TIGER-DICG configuration
│   │   └── envs/
│   │       ├── gather.yaml            # Gather environment config
│   │       └── gymma.yaml             # Gymma (Tag) environment config
│   ├── components/
│   │   ├── tiger_tgat.py              # TGAT: core temporal graph attention network
│   │   ├── tiger_graph.py             # NeighborFinder: temporal neighbor lookup
│   │   ├── attention_module.py        # Attention utilities
│   │   ├── gcn_module.py              # Graph convolution layers (used by DICG)
│   │   ├── episode_buffer.py          # Replay buffer
│   │   ├── epsilon_schedules.py       # Epsilon-greedy exploration schedules
│   │   ├── action_selectors.py        # Action selection strategies
│   │   └── transforms.py             # Observation transforms (e.g. one-hot)
│   ├── controllers/
│   │   ├── basic_controller.py        # BasicMAC (used by TIGER-MIX)
│   │   └── dicg_controller.py         # DICGraphMAC (used by TIGER-DICG)
│   ├── learners/
│   │   └── q_learner.py               # Q-learning with target networks and double-Q
│   ├── modules/
│   │   ├── agents/
│   │   │   └── rnn_agent.py           # GRU-based agent
│   │   └── mixers/
│   │       └── tiger_mixer.py         # TIGER QMIX mixer (TGAT integrated)
│   ├── runners/
│   │   ├── episode_runner.py          # Sequential episode runner
│   │   └── parallel_runner.py         # Parallel environment runner
│   ├── envs/
│   │   ├── gather.py                  # Gather cooperative task
│   │   ├── gymma.py                   # PettingZoo/Gymnasium wrapper (Tag)
│   │   ├── multiagentenv.py           # Base environment class
│   │   ├── pz_wrapper.py              # PettingZoo wrapper
│   │   ├── wrappers.py                # Observation wrappers
│   │   └── pretrained/                # Pretrained adversary agents for Tag
│   └── utils/
│       ├── logging.py
│       ├── rl_utils.py
│       └── timehelper.py
├── requirements.txt
└── install_dependencies.sh
```

## Environments

| Environment | Config key | Description |
|-------------|-----------|-------------|
| Gather | `gather` | 5 agents cooperatively collect moving targets on a grid |
| Tag | `gymma` with `env_args.key="pz-mpe-simple-tag-v3"` | Cooperative pursuit: agents tag an adversary |

## Results Logging

Results are saved to `results/tb_logs/` in TensorBoard format. Launch TensorBoard with:

```bash
tensorboard --logdir results/tb_logs
```

Sacred logs are written alongside and can be directed to a MongoDB observer by configuring `src/main.py`.

## Citation

If you use TIGER-MARL in your research, please cite:

```bibtex
@article{gupta2025tiger,
  title   = {TIGER-MARL: Enhancing Multi-Agent Reinforcement Learning with
             Temporal Information through Graph-based Embeddings and Representations},
  author  = {Gupta, Nikunj and Twardecka, Ludwika and Hare, James Zachary and
             Milzman, Jesse and Kannan, Rajgopal and Prasanna, Viktor},
  journal = {arXiv preprint arXiv:2511.08832},
  year    = {2025}
}
```

## Acknowledgements

This codebase builds upon [PyMARL](https://github.com/oxwhirl/pymarl) and incorporates ideas from [TGAT](https://arxiv.org/abs/2002.07962) (Temporal Graph Attention Networks) and [DICG](https://arxiv.org/abs/2006.11438) (Deep Implicit Coordination Graphs).

## License

See [LICENSE](LICENSE).
