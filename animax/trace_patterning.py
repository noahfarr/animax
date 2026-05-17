from __future__ import annotations

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

from animax._stimulus import is_active

_NO_ONSET = jnp.iinfo(jnp.int32).min // 2


@struct.dataclass
class TracePatterningState(environment.EnvState):
    trial_start: jax.Array
    cs_onsets: jax.Array
    us_onset: jax.Array
    distractor_onsets: jax.Array
    time: jax.Array


@struct.dataclass
class TracePatterningParams(environment.EnvParams):
    isi_min: int = 7
    isi_max: int = 13
    iti_min: int = 80
    iti_max: int = 129
    gamma: float = 0.9
    noise: float = 0.0
    us_activation_length: int = 2
    cs_activation_length: int = 4
    distractor_activation_length: int = 4
    max_steps_in_episode: int = 100000


def _produce_activation_patterns(
    seed: int, num_cs: int, num_activation_patterns: int
) -> jax.Array:
    indices = list(itertools.combinations(range(num_cs), num_cs // 2))
    if num_activation_patterns > len(indices):
        raise ValueError(
            f"num_activation_patterns={num_activation_patterns} exceeds the {len(indices)} available half-on patterns for num_cs={num_cs}."
        )
    rng = np.random.RandomState(seed)
    selected = rng.choice(len(indices), size=num_activation_patterns, replace=False)
    patterns = np.zeros((num_activation_patterns, num_cs), dtype=np.float32)
    for i, s in enumerate(selected):
        patterns[i, list(indices[s])] = 1.0
    return jnp.asarray(patterns)


def _binary_match(x: jax.Array, patterns: jax.Array) -> jax.Array:
    sum_x = jnp.sum(x)
    sum_1_x = jnp.sum(1.0 - x)
    safe_x = jnp.where(sum_x == 0, 1.0, sum_x)
    safe_1_x = jnp.where(sum_1_x == 0, 1.0, sum_1_x)
    ones_match = jnp.where(
        sum_x == 0, jnp.ones(patterns.shape[0]), jnp.floor(patterns @ x / safe_x)
    )
    zeros_match = jnp.where(
        sum_1_x == 0,
        jnp.ones(patterns.shape[0]),
        jnp.floor((1.0 - patterns) @ (1.0 - x) / safe_1_x),
    )
    return ones_match * zeros_match


class TracePatterning(
    environment.Environment[TracePatterningState, TracePatterningParams]
):
    def __init__(
        self,
        num_cs: int,
        num_activation_patterns: int,
        activation_patterns_prob: float = 0.5,
        num_distractors: int = 0,
        pattern_seed: int = 0,
    ):
        super().__init__()
        if num_cs < 2:
            raise ValueError("num_cs must be >= 2 for half-on patterns to exist.")
        self.num_cs = num_cs
        self.num_activation_patterns = num_activation_patterns
        self.activation_patterns_prob = activation_patterns_prob
        self.num_distractors = num_distractors
        self.obs_shape = (1 + num_cs + num_distractors,)
        self.activation_patterns = _produce_activation_patterns(
            pattern_seed, num_cs, num_activation_patterns
        )
        two_n = 2**num_cs
        denom = two_n - num_activation_patterns
        if denom == 0:
            self._p = jnp.float32(1.0)
        else:
            self._p = jnp.float32(
                (two_n * activation_patterns_prob - num_activation_patterns) / denom
            )

    @property
    def default_params(self) -> TracePatterningParams:
        return TracePatterningParams()

    @property
    def name(self) -> str:
        return "TracePatterning-v0"

    @property
    def num_actions(self) -> int:
        return 1

    def action_space(
        self, params: TracePatterningParams | None = None
    ) -> spaces.Discrete:
        return spaces.Discrete(1)

    def observation_space(self, params: TracePatterningParams) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: TracePatterningParams) -> spaces.Dict:
        big = params.max_steps_in_episode + params.isi_max + params.iti_max + 1
        return spaces.Dict(
            {
                "trial_start": spaces.Discrete(big),
                "cs_onsets": spaces.Box(-big, big, (self.num_cs,), jnp.int32),
                "us_onset": spaces.Discrete(big),
                "distractor_onsets": spaces.Box(
                    -big, big, (self.num_distractors,), jnp.int32
                ),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
            }
        )

    def _configure_trial(
        self, key: jax.Array, time: jax.Array, params: TracePatterningParams
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        k_use, k_pat, k_rand, k_isi, k_us, k_dist, k_iti = jax.random.split(key, 7)
        use_target = jax.random.uniform(k_use) < self._p
        target_idx = jax.random.randint(k_pat, (), 0, self.num_activation_patterns)
        target_pattern = self.activation_patterns[target_idx]
        random_pattern = jax.random.randint(k_rand, (self.num_cs,), 0, 2).astype(
            jnp.float32
        )
        cs_pattern = jnp.where(use_target, target_pattern, random_pattern)
        isi = jax.random.randint(k_isi, (), params.isi_min, params.isi_max + 1).astype(
            jnp.int32
        )
        iti = jax.random.randint(k_iti, (), params.iti_min, params.iti_max + 1).astype(
            jnp.int32
        )
        matches = jnp.sum(_binary_match(cs_pattern, self.activation_patterns)) > 0
        noise_roll = jax.random.uniform(k_us)
        us_fires = jnp.where(
            matches, noise_roll > params.noise, noise_roll < params.noise
        )
        if self.num_distractors > 0:
            distractor_pattern = jax.random.randint(
                k_dist, (self.num_distractors,), 0, 2
            ).astype(jnp.float32)
        else:
            distractor_pattern = jnp.zeros((0,), dtype=jnp.float32)
        return (cs_pattern, us_fires, isi, iti, distractor_pattern)

    def reset_env(
        self, key: jax.Array, params: TracePatterningParams
    ) -> tuple[jax.Array, TracePatterningState]:
        cs_pattern, us_fires, isi, iti, distractor_pattern = self._configure_trial(
            key, jnp.int32(0), params
        )
        cs_onsets = jnp.where(cs_pattern == 1, jnp.int32(0), jnp.int32(_NO_ONSET))
        us_onset = jnp.where(us_fires, isi, jnp.int32(_NO_ONSET)).astype(jnp.int32)
        distractor_onsets = jnp.where(
            distractor_pattern == 1, jnp.int32(0), jnp.int32(_NO_ONSET)
        )
        state = TracePatterningState(
            trial_start=(isi + iti).astype(jnp.int32),
            cs_onsets=cs_onsets,
            us_onset=us_onset,
            distractor_onsets=distractor_onsets,
            time=jnp.int32(0),
        )
        return (self.get_obs(state, params), state)

    def step_env(
        self,
        key: jax.Array,
        state: TracePatterningState,
        action: int | float | jax.Array,
        params: TracePatterningParams,
    ) -> tuple[jax.Array, TracePatterningState, jax.Array, jax.Array, dict[str, Any]]:
        del action
        new_time = state.time + 1
        is_trial = new_time == state.trial_start
        cs_pattern, us_fires, isi, iti, distractor_pattern = self._configure_trial(
            key, new_time, params
        )
        cs_fires = is_trial & (cs_pattern == 1)
        cs_onsets = jnp.where(cs_fires, new_time, state.cs_onsets).astype(jnp.int32)
        us_onset = jnp.where(
            is_trial & us_fires, new_time + isi, state.us_onset
        ).astype(jnp.int32)
        if self.num_distractors > 0:
            distractor_fires = is_trial & (distractor_pattern == 1)
            distractor_onsets = jnp.where(
                distractor_fires, new_time, state.distractor_onsets
            ).astype(jnp.int32)
        else:
            distractor_onsets = state.distractor_onsets
        trial_start = jnp.where(
            is_trial, new_time + isi + iti, state.trial_start
        ).astype(jnp.int32)
        new_state = TracePatterningState(
            trial_start=trial_start,
            cs_onsets=cs_onsets,
            us_onset=us_onset,
            distractor_onsets=distractor_onsets,
            time=new_time.astype(jnp.int32),
        )
        obs = self.get_obs(new_state, params)
        reward = is_active(
            new_state.time, new_state.us_onset, params.us_activation_length
        ).astype(jnp.float32)
        done = self.is_terminal(new_state, params)
        return (obs, new_state, reward, done, {"discount": jnp.float32(params.gamma)})

    def get_obs(
        self,
        state: TracePatterningState,
        params: TracePatterningParams | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del key
        if params is None:
            params = self.default_params
        us = is_active(state.time, state.us_onset, params.us_activation_length).astype(
            jnp.float32
        )
        cs = is_active(state.time, state.cs_onsets, params.cs_activation_length).astype(
            jnp.float32
        )
        parts = [us[None], cs]
        if self.num_distractors > 0:
            d = is_active(
                state.time, state.distractor_onsets, params.distractor_activation_length
            ).astype(jnp.float32)
            parts.append(d)
        return jnp.concatenate(parts)

    def is_terminal(
        self, state: TracePatterningState, params: TracePatterningParams
    ) -> jax.Array:
        return state.time >= params.max_steps_in_episode
