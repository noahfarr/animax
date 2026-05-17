from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from animax._stimulus import is_active

_NO_ONSET = jnp.iinfo(jnp.int32).min // 2


@struct.dataclass
class TraceConditioningState(environment.EnvState):
    trial_start: jax.Array
    cs_onset: jax.Array
    us_onset: jax.Array
    distractor_onsets: jax.Array
    time: jax.Array


@struct.dataclass
class TraceConditioningParams(environment.EnvParams):
    isi_min: int = 7
    isi_max: int = 13
    iti_min: int = 80
    iti_max: int = 129
    gamma: float = 0.9
    us_activation_length: int = 2
    cs_activation_length: int = 4
    distractor_activation_length: int = 4
    max_steps_in_episode: int = 100000


class TraceConditioning(
    environment.Environment[TraceConditioningState, TraceConditioningParams]
):
    def __init__(self, num_distractors: int = 0):
        super().__init__()
        if not 0 <= num_distractors <= 10:
            raise ValueError(
                "num_distractors must be in [0, 10] to match the source benchmark."
            )
        self.num_distractors = num_distractors
        self.obs_shape = (2 + num_distractors,)
        if num_distractors > 0:
            self._distractor_probs = 1.0 / jnp.arange(
                10, 10 * num_distractors + 10, 10, dtype=jnp.float32
            )
        else:
            self._distractor_probs = jnp.zeros(0, dtype=jnp.float32)

    @property
    def default_params(self) -> TraceConditioningParams:
        return TraceConditioningParams()

    @property
    def name(self) -> str:
        return "TraceConditioning-v0"

    @property
    def num_actions(self) -> int:
        return 1

    def action_space(
        self, params: TraceConditioningParams | None = None
    ) -> spaces.Discrete:
        return spaces.Discrete(1)

    def observation_space(self, params: TraceConditioningParams) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: TraceConditioningParams) -> spaces.Dict:
        big = params.max_steps_in_episode + params.isi_max + params.iti_max + 1
        return spaces.Dict(
            {
                "trial_start": spaces.Discrete(big),
                "cs_onset": spaces.Discrete(big),
                "us_onset": spaces.Discrete(big),
                "distractor_onsets": spaces.Box(
                    -big, big, (self.num_distractors,), jnp.int32
                ),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
            }
        )

    def reset_env(
        self, key: jax.Array, params: TraceConditioningParams
    ) -> tuple[jax.Array, TraceConditioningState]:
        k_isi, k_iti, k_dist = jax.random.split(key, 3)
        isi = jax.random.randint(k_isi, (), params.isi_min, params.isi_max + 1)
        iti = jax.random.randint(k_iti, (), params.iti_min, params.iti_max + 1)
        if self.num_distractors > 0:
            rolls = jax.random.uniform(k_dist, (self.num_distractors,))
            fire = rolls < self._distractor_probs
            distractor_onsets = jnp.where(fire, jnp.int32(0), jnp.int32(_NO_ONSET))
        else:
            distractor_onsets = jnp.zeros((0,), dtype=jnp.int32)
        state = TraceConditioningState(
            trial_start=(isi + iti).astype(jnp.int32),
            cs_onset=jnp.int32(0),
            us_onset=isi.astype(jnp.int32),
            distractor_onsets=distractor_onsets,
            time=jnp.int32(0),
        )
        return (self.get_obs(state, params), state)

    def step_env(
        self,
        key: jax.Array,
        state: TraceConditioningState,
        action: int | float | jax.Array,
        params: TraceConditioningParams,
    ) -> tuple[jax.Array, TraceConditioningState, jax.Array, jax.Array, dict[str, Any]]:
        del action
        new_time = state.time + 1
        k_isi, k_iti, k_dist = jax.random.split(key, 3)
        isi = jax.random.randint(k_isi, (), params.isi_min, params.isi_max + 1).astype(
            jnp.int32
        )
        iti = jax.random.randint(k_iti, (), params.iti_min, params.iti_max + 1).astype(
            jnp.int32
        )
        is_trial = new_time == state.trial_start
        cs_onset = jnp.where(is_trial, new_time, state.cs_onset).astype(jnp.int32)
        us_onset = jnp.where(is_trial, new_time + isi, state.us_onset).astype(jnp.int32)
        trial_start = jnp.where(
            is_trial, new_time + isi + iti, state.trial_start
        ).astype(jnp.int32)
        if self.num_distractors > 0:
            prev_active = is_active(
                state.time, state.distractor_onsets, params.distractor_activation_length
            )
            rolls = jax.random.uniform(k_dist, (self.num_distractors,))
            fire = ~prev_active & (rolls < self._distractor_probs)
            distractor_onsets = jnp.where(
                fire, new_time, state.distractor_onsets
            ).astype(jnp.int32)
        else:
            distractor_onsets = state.distractor_onsets
        new_state = TraceConditioningState(
            trial_start=trial_start,
            cs_onset=cs_onset,
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
        state: TraceConditioningState,
        params: TraceConditioningParams | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del key
        if params is None:
            params = self.default_params
        us = is_active(state.time, state.us_onset, params.us_activation_length).astype(
            jnp.float32
        )
        cs = is_active(state.time, state.cs_onset, params.cs_activation_length).astype(
            jnp.float32
        )
        head = jnp.stack([us, cs])
        if self.num_distractors == 0:
            return head
        d = is_active(
            state.time, state.distractor_onsets, params.distractor_activation_length
        ).astype(jnp.float32)
        return jnp.concatenate([head, d])

    def is_terminal(
        self, state: TraceConditioningState, params: TraceConditioningParams
    ) -> jax.Array:
        return state.time >= params.max_steps_in_episode
