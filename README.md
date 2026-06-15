# Call Break RL Agent

A reinforcement learning agent for the card game **Call Break** using PPO with self-play,
policy/value networks, and a curriculum training pipeline designed to beat heuristic baselines.

## Game Overview
Call Break is a 4-player trick-taking card game where spades are always trump.
Each player bids how many tricks they will win. Scoring rewards meeting or exceeding
your bid; failing to meet it incurs a penalty.

## Project Structure
```
callbreak_rl/
├── config/             # YAML configuration files
├── environment/        # Call Break game engine (Gym-compatible)
│   ├── card.py         # Card and Suit enums
│   ├── deck.py         # Deck creation and shuffling
│   ├── trick.py        # Single-trick logic
│   ├── round.py        # Full round (13 tricks + bidding)
│   └── callbreak_env.py# OpenAI Gym environment wrapper
├── agents/
│   ├── base_agent.py   # Abstract agent interface
│   ├── random_agent.py # Uniform random baseline
│   ├── heuristic/      # Rule-based baselines
│   │   ├── safe_bet_agent.py
│   │   ├── safe_play_agent.py
│   │   └── basic_bet_agent.py
│   └── rl/             # RL-based agents
│       ├── ppo_agent.py
│       ├── bid_agent.py
│       └── play_agent.py
├── models/             # Neural network components
│   ├── encoder.py      # State encoder (MLP / Transformer)
│   ├── policy_network.py
│   ├── value_network.py
│   └── actor_critic.py # Combined actor-critic head
├── training/
│   ├── trainer.py      # PPO training loop
│   ├── self_play.py    # Self-play pool manager
│   ├── curriculum.py   # Opponent curriculum scheduler
│   └── replay_buffer.py# Rollout buffer
├── evaluation/
│   ├── evaluator.py    # Head-to-head match runner
│   └── elo.py          # Elo rating tracker
├── utils/
│   ├── state_encoder.py# Convert game state → tensor
│   ├── reward_shaper.py# Reward shaping logic
│   ├── logger.py       # Structured logging + TensorBoard
│   └── metrics.py      # Win-rate, bid accuracy, etc.
├── scripts/
│   ├── train.py        # Entry point: training
│   ├── evaluate.py     # Entry point: evaluation
│   └── play.py         # Entry point: interactive play
└── tests/              # Unit tests
```

## Quick Start
```bash
pip install -r requirements.txt

# Train from scratch vs random opponents
python scripts/train.py --config config/config.yaml --stage random

# Evaluate against all baselines
python scripts/evaluate.py --checkpoint checkpoints/best.pt

# Play interactively
python scripts/play.py
```

## Training Curriculum
| Stage | Opponents          | Goal                          |
|-------|--------------------|-------------------------------|
| 1     | 3× Random          | Learn card hierarchy          |
| 2     | 2× Random + 1 Safe | Exploit conservative players  |
| 3     | 3× Safe Play       | Compete at table level        |
| 4     | Self-play pool     | Discover emergent strategies  |

## Key Design Decisions
- **Separate bid and play policies** — different information sets and objectives
- **Transformer encoder** — attention over card play history captures voiding/counting
- **PPO + self-play** — stable multi-agent training without full CFR complexity
- **Reward shaping** — intermediate trick rewards reduce sparse-reward problem
