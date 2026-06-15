"""
callbreak_env.py
----------------
OpenAI Gymnasium-compatible environment for Call Break.

Observation space  — flat numpy array (see _encode_observation)
Action space       — Discrete(52) for play, Discrete(13) for bid
Reward             — shaped per-trick + final round score
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.round import Round
from environment.card import Card
from utils.state_encoder import StateEncoder
from utils.reward_shaper import RewardShaper


# Observation vector length (see StateEncoder for breakdown):
#   52  cards in hand (binary)
#   52  cards played globally (binary)
#   52  cards in current trick (binary, positional 4×13)
#   4   bids normalised [0,1]
#   4   tricks won normalised [0,1]
#   4   one-hot position of current trick leader
#   1   tricks remaining normalised
#   1   bid deficit normalised
#   4   one-hot current player position at table
# -----
# 174  total
OBS_DIM = 174
NUM_CARDS = 52
NUM_BID_ACTIONS = 13  # bids 1-13


class CallBreakEnv(gym.Env):
    """
    Single-agent view of a 4-player Call Break game.
    The three opponents are controlled by agents passed in at construction.
    The learning agent always occupies seat 0.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        opponent_agents: List[Any],    # 3 agents implementing BaseAgent
        num_rounds: int = 5,
        reward_shaping: bool = True,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert len(opponent_agents) == 3, "Need exactly 3 opponents."
        self.opponent_agents = opponent_agents
        self.num_rounds = num_rounds
        self.render_mode = render_mode

        self.encoder = StateEncoder()
        self.reward_shaper = RewardShaper(enabled=reward_shaping)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        # Action space switches between bidding (13) and playing (52).
        # The environment handles masking; the agent sees only legal actions.
        self.action_space = spaces.Discrete(NUM_CARDS)

        self._rng = np.random.default_rng(seed)
        self._round: Optional[Round] = None
        self._round_number: int = 0
        self._total_scores: List[float] = [0.0] * 4
        self._phase: str = "bid"   # "bid" | "play"

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Start a new game (reset round counter and scores).
        Deal cards, run opponents' bids, return agent's first observation.
        Returns (observation, info_dict).
        """
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Agent takes `action` (card index or bid index).
        After the agent acts, advance the game until it is the agent's
        turn again (running opponent actions automatically).
        Returns (observation, reward, terminated, truncated, info).

        `terminated` = True when all num_rounds are complete.
        `info` includes: {
            'legal_actions': [...],
            'phase': 'bid' | 'play',
            'tricks_won': [...],
            'bids': [...],
            'scores': [...],
            'round_number': int,
        }
        """
        raise NotImplementedError

    def _get_observation(self) -> np.ndarray:
        """Encode the current game state for the learning agent (player 0)."""
        raise NotImplementedError

    def _get_legal_action_mask(self) -> np.ndarray:
        """
        Return a boolean mask of shape (52,) or (13,) over the action space
        indicating which actions are currently legal.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal game flow
    # ------------------------------------------------------------------

    def _start_new_round(self) -> None:
        """Initialise a new Round, deal cards, begin bidding phase."""
        raise NotImplementedError

    def _run_opponents_until_agent_turn(self) -> float:
        """
        Step through opponent actions until it is player 0's turn.
        Accumulate and return any intermediate shaped rewards earned by
        player 0 during opponent tricks (e.g. tricks won against player 0).
        """
        raise NotImplementedError

    def _apply_agent_bid(self, bid: int) -> None:
        """Record the agent's bid (action 0 → bid 1, ..., action 12 → bid 13)."""
        raise NotImplementedError

    def _apply_agent_play(self, card_index: int) -> float:
        """
        Play the card corresponding to `card_index` for player 0.
        Returns intermediate shaped reward (0.0 if no trick completed).
        Raises ValueError if the card is not in hand or not a legal play.
        """
        raise NotImplementedError

    def _round_end_reward(self) -> float:
        """Compute and return player 0's score for the just-completed round."""
        raise NotImplementedError

    def _advance_to_next_round(self) -> None:
        """Finalise scores, increment round counter, start a new round."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        """
        Render the current game state.
        'human' mode: print to stdout.
        'ansi' mode: return a string.
        """
        raise NotImplementedError

    def close(self) -> None:
        pass
