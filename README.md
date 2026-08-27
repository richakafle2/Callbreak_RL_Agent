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
python -m scripts.train --config config/config.yaml --stage random

# Continue training a specific stage from a checkpoint
python -m scripts.train --config config/config.yaml --stage mixed

# Evaluate against all baselines
python -m scripts.evaluate --checkpoint checkpoints_mlp2/best.pt

# Play interactively
python scripts/play.py
```

## Training Curriculum
| Stage        | Opponents                          | Advance threshold | Goal                          |
|--------------|-------------------------------------|--------------------|--------------------------------|
| `random`     | 3× Random                           | 70% win rate        | Learn card hierarchy          |
| `mixed`      | 2× Random + 1× Safe Play            | 65% win rate        | Exploit conservative players  |
| `safe`       | 3× Safe Play                        | 60% win rate        | Compete at table level        |
| `mixed_self` | `["basic_bet", "self", "safe_play"]`| **55%** win rate     | Bridge into self-play against tougher, more varied opponents |
| `self_play`  | Self-play pool                      | n/a (fixed timesteps)| Discover emergent strategies  |

> **Note on `mixed_self`:** originally set at 0.65 like the other stages, but lowered to **0.55**
> after analysis showed that in a 4-player winner-take-all format, 65% is structurally
> unreachable once opponents are comparably skilled — no single seat can win two-thirds
> of games when all four seats are playing well. Curriculum advancement is gated on a
> rolling window of *current-stage* results only; an earlier bug that evaluated against
> the global eval suite instead caused the agent to silently never leave the `mixed` stage
> across a full 10M-step run.

## Key Design Decisions
- **Separate bid and play policies** — different information sets and objectives
- **Transformer encoder** — attention over card play history captures voiding/counting
- **PPO + self-play** — stable multi-agent training without full CFR complexity
- **Reward shaping** — intermediate trick rewards reduce sparse-reward problem, subject to
  the policy-invariance caveats below

---

## Evaluation Metrics — What Each Thing Means

Running `python -m scripts.evaluate --checkpoint <ckpt>` produces a summary block plus
three detail tables. Here's what each field means and how to read it.

### Top-line summary
| Metric | Meaning |
|---|---|
| **Overall win rate** | Fraction of games where the agent (seat 0) finishes with the highest total score, averaged across every opponent matchup in the eval suite. |
| **Elo rating** | Running Elo estimate of agent strength, updated pairwise (agent vs. each opponent) after every game via the standard logistic expected-score formula, K-factor from `config.yaml → evaluation.elo.k_factor`. Useful for comparing checkpoints across training runs even when the opponent mix changes. |
| **Bid accuracy** | Fraction of *rounds* (not games) where the agent won ≥ its declared bid. This is the single best proxy for "is the bidding head calibrated," independent of who else is at the table. |
| **Avg overtrick** | Mean number of tricks won *above* bid, counted only in rounds where the bid was met. High values mean the agent is bidding conservatively relative to what it can actually take — it's leaving expected score on the table by under-bidding hands it then dominates. |

### Per-opponent table (`Win rate`, `Avg score`)
One row per baseline (`random`, `safe_bet`, `safe_play`, `basic_bet`). `Win rate` is the
agent's win rate specifically in games against that opponent mix (3 seats of that heuristic);
`Avg score` is the agent's mean total game score in that matchup. Because `basic_bet` is the
strongest heuristic, its row is the standard headline number when reporting progress
(currently the binding constraint — see Strategies below).

### Bid comparison table
Columns, per matchup and per seat (`PPO` vs. the named heuristic):

| Column | Meaning |
|---|---|
| **Avg bid** | Mean declared bid across all rounds for that seat. |
| **Avg tricks** | Mean tricks actually won across those rounds. |
| **Avg gap** | `Avg tricks − Avg bid`. Positive = seat tends to over-perform its bid (sandbagging / under-confidence); negative = seat tends to under-perform (over-bidding, risk of bust). |
| **Bid-strength corr (r)** | Pearson correlation between the seat's declared bid and an independent hand-strength heuristic (high-card points + 1.5× trump count) computed by the evaluator. This answers "does the bid actually track hand quality?" independent of whether the *level* of the bid is well-calibrated. PPO's `r` sits around **+0.84–0.86** across all matchups — strong and consistent. Heuristic opponents vary a lot here: `basic_bet` (+0.78) and `safe_bet` (+0.77) track hand strength reasonably well; `random` (−0.01) and `safe_play` (+0.02) essentially don't (expected — `random` bids arbitrarily and `safe_play`'s conservative rule flattens the signal). |

**How to use this table:** a large positive `Avg gap` combined with high `Bid-strength corr`
means the model *knows* how strong its hand is but is bidding too low anyway — a bid-calibration
problem, fixable via `overtrick_bonus_rate` / `underbid_penalty_rate` in `reward_shaper.py`.
A small `Avg gap` with a comparable or better gap than the opponent, but a win-rate deficit
that persists, points at play-phase quality instead (see `basic_bet` note below).

### Play-phase diagnostics table
Columns, per matchup and seat:

| Column | Meaning |
|---|---|
| **TrickWin%** | Fraction of all tricks played that this seat wins outright — the base rate of "does this agent take tricks," across both contested and uncontested situations. |
| **PreTrump%** | Fraction of tricks won *before* trump has been established as the controlling suit for that trick (i.e., won by following suit / natural high card, no spade needed). |
| **PostTrump%** | Fraction of tricks won *after* a trump has been played into the trick (i.e., the trick required a spade to take). Comparing Pre vs. Post trump-win share indicates *when* in a round a seat relies on trump — spending trumps early vs. holding them for critical late tricks. |
| **PostWin%** | Win rate on tricks contested specifically against opponents who still need tricks to make their own bid — the "denial" situations. Lower than `TrickWin%` generally means the agent isn't prioritizing denial plays as much as raw trick-taking. |
| **AvgNeedy** | Average number of opponents at the table who are still short of their bid at the moment a trick is led — a proxy for how much genuine denial opportunity exists in a given matchup/seat's rounds, used to sanity-check whether `PostWin%` differences reflect skill or just differing opportunity. |

> These four columns were added specifically to investigate the persistent `basic_bet` win-rate
> gap once bid-calibration was ruled out as the driver — see **Strategies & Improvements** below.
> Definitions above reflect the diagnostic's intent as designed; if `evaluator.py` has since
> changed the exact computation, treat this table as a guide to re-derive from source rather
> than a guaranteed spec.

### Tercile breakdown (weak / medium / strong hand strength)
The same bid-comparison columns (`AvgBid`, `AvgTricks`, `AvgGap`, `AvgStr`, `N`), bucketed by
the evaluator's hand-strength heuristic split into thirds. This separates two failure modes that
average-over-all-hands metrics blur together:
- A **bid-calibration** problem shows up as `AvgGap` shrinking (toward 0 or negative) specifically
  in the **strong** bucket — i.e., the agent is fine on so-so hands but doesn't recognize a great hand.
- A **structural/opponent** difference shows up as gaps that move together across buckets between
  PPO and the heuristic, rather than diverging at one end.
`N` is the sample size for that bucket — with several hundred to over a thousand rounds per
cell in typical eval runs, these breakdowns are trustworthy rather than noise.

---

## Strategies & Improvements Explored

A running log of the diagnostic work and fixes applied while chasing the `basic_bet`
win-rate gap (the hardest matchup — PPO wins that seat only ~30-37% of the time despite
dominating `random` and `safe_play`).

### Fixes applied
- **Bid calibration retune** — `overtrick_bonus_rate` 0.1 → 0.3 and `underbid_penalty_rate`
  1.0 → 0.6 in `reward_shaper.py`, plus replacing a hardcoded `reward = 0.0` for overtricks
  in `trick_won_reward()` with `self.trick_reward * self.overtrick_bonus_rate`. This reliably
  raised bid comparisons (8/8 avg-bid comparisons improved) with bid-strength correlation
  holding steady at 0.83–0.86.
- **Curriculum advancement bug** — `should_advance()` was checking cumulative win rate since
  the stage began, so stale early-stage data could permanently suppress advancement; also,
  `get_opponents()` for self-play stages was ignoring the configured opponent list and
  replacing all seats uniformly. Both fixed; `timesteps_in_stage` (previously never
  incremented, so the terminal fixed-timestep stage could never complete) was also fixed.
- **`mixed_self` bridge stage added** — a new stage (`["basic_bet", "self", "safe_play"]`)
  between `safe` and `self_play`, easing the transition into tougher/self-play opponents.
  Threshold set to 0.55 (see curriculum table note above) after 0.65 proved structurally
  unreachable.
- **`is_self_play` bug** — was checking `stage.name == "self_play"` literally instead of
  inspecting opponent list contents, causing a `KeyError: 'self'` once `mixed_self` existed.
- **Denial bonus added** — `trick_won_reward()` now rewards winning a trick that opponents
  still need (wired via a new `_opponent_bid_tricks_state()` helper in `callbreak_env.py`),
  intended to encourage denial play, which was previously unrewarded (`final_game_reward()`
  existed in `reward_shaper.py` but was confirmed dead code, never called).
- **Import path fix** — `scripts/play.py` had `ModuleNotFoundError` for
  `agents.basic_bet_agent`; corrected to `agents.heuristic.basic_bet_agent`. Other
  `from agents import ...` lines in the same file likely need the same subpackage fix.

### Known issues / open investigations
- **Reward channel ambiguity (unresolved):** `Round.calculate_scores()` uses its own
  hardcoded scoring constants (`0.1` overtrick rate, full `-bid` underbid penalty),
  completely independent of `RewardShaper.round_score()`. Whether config retunes have
  actually been affecting PPO's training signal depends on whether `trainer.py` uses
  `round.scores` directly (in which case retunes were **inert** on the dominant terminal
  reward channel) or calls `reward_shaper.round_score()` separately (in which case retunes
  were working as intended). Still awaiting the relevant `trainer.py` section to confirm.
  **Important constraint regardless of outcome:** `Round.calculate_scores()` must stay a
  fixed, non-tunable rule — it's also almost certainly what `evaluator.py` uses to
  determine wins for win-rate reporting, and routing it through shaper hyperparameters
  would make "win rate" partially measure your hyperparameter choice instead of policy
  quality, corrupting every cross-experiment comparison.
- **Non-determinism confirmed:** identical config + seed produces materially different
  outcomes (e.g. `basic_bet` win rate 33.5% vs. 37.5% run-to-run, ~41 Elo swing). Suspected
  causes: missing `torch.use_deterministic_algorithms(True)`, non-deterministic cuDNN ops,
  or parallel envs not seeded as `seed + env_idx`. Until fixed, single-run win-rate deltas
  aren't reliable evidence for or against a change — establish a 2–3-run noise floor per
  condition, and prefer the bid-diagnostic tables (more stable across runs) over raw win
  rate when judging an experiment.
- **Denial bonus urgency-weighting flaw:** the "needy opponent" condition (`tricks_won < bid`)
  is true for nearly every opponent at the start of nearly every round, not just when
  they're genuinely at risk of missing their bid — so the bonus currently rewards early,
  broad trick-grabbing rather than targeted denial. Needs gating on something closer to
  `tricks_won <= bid` **and** few tricks remaining, before its effect can be trusted.
- **`shaped_reward()` possibly dead code** — grep shows the potential-based shaping method
  is never called outside its own definition; worth confirming and either wiring it in or
  removing it.
- **Potential-function policy-invariance bug** — `Φ(s) = tricks_won / bid` gives nonzero
  terminal potential (breaking the policy-invariance guarantee of potential-based shaping)
  and quietly overweights low-bid rounds in PPO's advantage estimates, since the same
  absolute trick difference produces a much larger ratio swing at bid=1 than at bid=8.

### Diagnostic conclusion so far
Once PPO's bids were brought roughly in line with `basic_bet`'s own (~2.44 vs. ~2.64 average),
the win-rate gap against it persisted — evidence that **the remaining `basic_bet` gap is a
play-phase problem (card selection / denial), not a bidding problem**. This is the opposite
conclusion for `safe_bet`, where PPO already both out-bids and out-tricks the heuristic at
every hand-strength tercile, meaning the bid retune is expected to help `basic_bet` far more
than `safe_bet`.

### Experimental discipline established
- **Single-variable isolation:** change one thing at a time; when comparing two configs,
  resume both from the same checkpoint with the same seed rather than comparing against
  historical runs with different training lineages.
- **Bid diagnostics over win rate:** given the confirmed non-determinism, average bid / average
  gap are more stable signals across runs than raw win rate.
- **Planned controlled experiment (not yet run):** same seed, same resume checkpoint,
  `denial_bonus=0.05` vs. `denial_bonus=0.0`, to properly isolate the denial bonus's effect
  once the urgency-weighting flaw above is fixed.

### Other directions explored / on the roadmap
- **Architecture comparison:** Transformer encoder (Elo 1019, 64.1% overall) vs. MLP encoder
  (Elo 1012, 62% overall) — a modest edge for the Transformer so far, not yet conclusive
  given the non-determinism caveat above.
- **Rank-based payout structure** (1st: 0, 2nd: −1, 3rd: −2, 4th: −3) discussed as a possible
  reward redesign — not yet reflected in the reward function.
- **MCTS** as a possible complement/alternative to pure PPO self-play — discussed, not started.
- **`scripts/interactive_play.py`** — a human-playable advisor mode that takes seat number,
  other players' bids/throws, and the round-in-progress state, and returns the trained
  policy's bid/play recommendation. Blocked on sharing `actor_critic.py`, `encoder.py`, and
  the checkpoint-loading pattern so the integration matches the real `ActorCritic` interface
  rather than guessing at conventions.

---

## Best Model — Evaluation Summary

Results below are from the strongest checkpoint evaluated to date.

```
Overall win rate     69.1%
Elo rating           1070
Bid accuracy         92.8%
Avg overtrick        1.28
```

| Opponent   | Win rate | Avg score |
|------------|----------|-----------|
| random     | 100.0%   | 12.25     |
| safe_bet   | 60.0%    | 11.75     |
| safe_play  | 87.0%    | 9.34      |
| **basic_bet** | **29.5%** | 11.00 |

`basic_bet` remains the clear outstanding target — every other matchup is comfortably won,
consistent with the play-phase (rather than bidding) diagnosis above.

### Bid comparison by matchup
| Matchup | Seat | Avg bid | Avg tricks | Avg gap | Bid-str r |
|---|---|---|---|---|---|
| random | PPO | 2.43 | 3.94 | +1.51 | +0.84 |
| random | Random | 7.08 | 3.02 | −4.06 | −0.01 |
| safe_bet | PPO | 2.40 | 3.68 | +1.28 | +0.86 |
| safe_bet | SafeBet | 1.70 | 3.11 | +1.41 | +0.77 |
| safe_play | PPO | 2.44 | 3.14 | +0.69 | +0.84 |
| safe_play | SafePlay | 2.23 | 3.29 | +1.06 | +0.02 |
| basic_bet | PPO | 2.44 | 3.41 | +0.97 | +0.85 |
| basic_bet | BasicBet | 2.64 | 3.20 | +0.55 | +0.78 |

PPO's bid-strength correlation is strong and consistent (0.84–0.86) across every matchup;
its avg gap vs. `basic_bet` (+0.97) is now noticeably larger than `basic_bet`'s own (+0.55),
meaning PPO is if anything sandbagging slightly relative to the heuristic here — reinforcing
that bidding is not the current bottleneck against this opponent.

### Play-phase diagnostics by matchup
| Matchup | Seat | TrickWin% | PreTrump% | PostTrump% | PostWin% | AvgNeedy |
|---|---|---|---|---|---|---|
| random | PPO | 30.3% | 55.2% | 56.4% | 23.4% | 2.63 |
| random | Random | 23.2% | 61.5% | 59.3% | 26.7% | 2.05 |
| safe_bet | PPO | 28.3% | 47.7% | 58.7% | 19.2% | 0.63 |
| safe_bet | SafeBet | 23.9% | 51.3% | 51.7% | 19.3% | 0.41 |
| safe_play | PPO | 24.2% | 62.0% | 58.1% | 19.2% | 0.75 |
| safe_play | SafePlay | 25.3% | 82.6% | 48.0% | 26.0% | 1.06 |
| basic_bet | PPO | 26.2% | 53.3% | 60.0% | 17.1% | 1.57 |
| basic_bet | BasicBet | 24.6% | 55.6% | 58.9% | 18.5% | 1.30 |

Against `basic_bet` specifically: PPO's overall `TrickWin%` (26.2%) is actually slightly
*higher* than BasicBet's own (24.6%), but its `PostWin%` — the win rate specifically in
denial situations against opponents who still need tricks — is *lower* (17.1% vs. 18.5%),
despite facing more denial opportunity on average (`AvgNeedy` 1.57 vs. 1.30). This is the
clearest evidence yet that the remaining gap is about **prioritizing denial plays**, not raw
trick-taking ability — directly motivating the (not-yet-cleanly-tested) denial bonus
experiment above.

### Bid comparison by hand-strength tercile
| Matchup | Seat | Bucket | AvgBid | AvgTricks | AvgGap | AvgStr | N |
|---|---|---|---|---|---|---|---|
| random | PPO | weak | 1.54 | 2.61 | +1.07 | 10.12 | 325 |
| random | PPO | medium | 2.41 | 3.89 | +1.47 | 14.89 | 381 |
| random | PPO | strong | 3.44 | 5.47 | +2.03 | 20.12 | 294 |
| random | Random | weak | 7.11 | 1.76 | −5.36 | 9.98 | 1026 |
| random | Random | medium | 7.02 | 3.00 | −4.02 | 14.94 | 1056 |
| random | Random | strong | 7.11 | 4.46 | −2.65 | 20.28 | 918 |
| safe_bet | PPO | weak | 1.53 | 2.58 | +1.05 | 10.20 | 390 |
| safe_bet | PPO | medium | 2.40 | 3.73 | +1.33 | 15.23 | 316 |
| safe_bet | PPO | strong | 3.56 | 5.09 | +1.53 | 20.34 | 294 |
| safe_bet | SafeBet | weak | 1.07 | 1.90 | +0.83 | 10.17 | 1121 |
| safe_bet | SafeBet | medium | 1.57 | 3.13 | +1.56 | 15.25 | 965 |
| safe_bet | SafeBet | strong | 2.60 | 4.56 | +1.96 | 20.36 | 914 |
| safe_play | PPO | weak | 1.55 | 1.93 | +0.38 | 10.22 | 374 |
| safe_play | PPO | medium | 2.51 | 3.22 | +0.72 | 15.28 | 317 |
| safe_play | PPO | strong | 3.47 | 4.52 | +1.06 | 20.14 | 309 |
| safe_play | SafePlay | weak | 2.15 | 2.11 | −0.05 | 10.29 | 1131 |
| safe_play | SafePlay | medium | 2.24 | 3.34 | +1.09 | 15.21 | 990 |
| safe_play | SafePlay | strong | 2.32 | 4.75 | +2.44 | 20.37 | 879 |
| basic_bet | PPO | weak | 1.48 | 2.22 | +0.74 | 9.99 | 328 |
| basic_bet | PPO | medium | 2.41 | 3.38 | +0.97 | 14.92 | 360 |
| basic_bet | PPO | strong | 3.49 | 4.70 | +1.21 | 20.14 | 312 |
| basic_bet | BasicBet | weak | 1.73 | 2.12 | +0.39 | 9.86 | 1021 |
| basic_bet | BasicBet | medium | 2.63 | 3.20 | +0.57 | 14.96 | 1086 |
| basic_bet | BasicBet | strong | 3.71 | 4.43 | +0.72 | 20.44 | 893 |

Against `basic_bet`, PPO's `AvgGap` grows steadily from weak → strong (+0.74 → +1.21) while
BasicBet's own grows more slowly (+0.39 → +0.72) — PPO is sandbagging relatively more on its
best hands, the opposite of a classic "doesn't recognize a strong hand" bid-calibration bug.
Combined with the play-phase table above, this points squarely at **in-trick decision quality
(denial play)**, not bidding, as the lever most likely to close the remaining gap.
