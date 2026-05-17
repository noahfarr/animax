from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from classical_conditioning_benchmarks import NoisyPatterning as NPNoisyPatterning
from classical_conditioning_benchmarks import TraceConditioning as NPTraceConditioning
from classical_conditioning_benchmarks import TracePatterning as NPTracePatterning

import animax

N_STEPS = 4000
N_SEEDS = 16
TOL = 0.02


def _np_rollout_trace_conditioning(
    seed: int, n_steps: int, num_distractors: int
) -> np.ndarray:
    env = NPTraceConditioning(
        seed=seed,
        ISI_interval=(7, 13),
        ITI_interval=(80, 129),
        gamma=0.9,
        num_distractors=num_distractors,
        activation_lengths={"CS": 4, "US": 2, "distractor": 4},
    )
    env.reset()
    obs_seq = np.zeros((n_steps + 1, 2 + num_distractors), dtype=np.float32)
    obs_seq[0] = env.observation()
    for t in range(n_steps):
        _, _, _, obs = env.step(None)
        obs_seq[t + 1] = obs
    return obs_seq


def _jax_rollout_trace_conditioning(
    key, n_steps: int, num_distractors: int
) -> np.ndarray:
    env = animax.TraceConditioning(num_distractors=num_distractors)
    params = env.default_params

    @functools.partial(jax.jit, static_argnames=("n",))
    def go(key, n):
        obs0, state = env.reset(key, params)

        def body(carry, _):
            state, key = carry
            key, k = jax.random.split(key)
            obs, state, _, _, _ = env.step(k, state, 0, params)
            return ((state, key), obs)

        (_, _), seq = jax.lax.scan(body, (state, key), None, length=n)
        return jnp.concatenate([obs0[None], seq], axis=0)

    return np.asarray(go(key, n_steps))


@pytest.mark.parametrize("num_distractors", [0, 2])
def test_trace_conditioning_mean_activity_matches(num_distractors: int):
    np_runs = np.stack(
        [
            _np_rollout_trace_conditioning(s, N_STEPS, num_distractors)
            for s in range(N_SEEDS)
        ]
    )
    jax_runs = np.stack(
        [
            _jax_rollout_trace_conditioning(
                jax.random.PRNGKey(s), N_STEPS, num_distractors
            )
            for s in range(N_SEEDS)
        ]
    )
    np_mean = np_runs.mean(axis=(0, 1))
    jax_mean = jax_runs.mean(axis=(0, 1))
    assert np_mean.shape == jax_mean.shape
    np.testing.assert_allclose(jax_mean, np_mean, atol=TOL)


def _np_rollout_trace_patterning(seed: int, n_steps: int) -> np.ndarray:
    env = NPTracePatterning(
        seed=seed,
        ISI_interval=(7, 13),
        ITI_interval=(80, 129),
        gamma=0.9,
        num_CS=4,
        num_activation_patterns=3,
        activation_patterns_prob=0.5,
        num_distractors=2,
        activation_lengths={"CS": 4, "US": 2, "distractor": 4},
        noise=0.0,
    )
    env.reset()
    obs_seq = np.zeros((n_steps + 1, 1 + 4 + 2), dtype=np.float32)
    obs_seq[0] = env.observation()
    for t in range(n_steps):
        _, _, _, obs = env.step(None)
        obs_seq[t + 1] = obs
    return obs_seq


def _jax_rollout_trace_patterning(key, n_steps: int) -> np.ndarray:
    env = animax.TracePatterning(
        num_cs=4,
        num_activation_patterns=3,
        activation_patterns_prob=0.5,
        num_distractors=2,
        pattern_seed=0,
    )
    params = env.default_params

    @functools.partial(jax.jit, static_argnames=("n",))
    def go(key, n):
        obs0, state = env.reset(key, params)

        def body(carry, _):
            state, key = carry
            key, k = jax.random.split(key)
            obs, state, _, _, _ = env.step(k, state, 0, params)
            return ((state, key), obs)

        (_, _), seq = jax.lax.scan(body, (state, key), None, length=n)
        return jnp.concatenate([obs0[None], seq], axis=0)

    return np.asarray(go(key, n_steps))


def test_trace_patterning_mean_activity_matches():
    np_runs = np.stack(
        [_np_rollout_trace_patterning(s, N_STEPS) for s in range(N_SEEDS)]
    )
    jax_runs = np.stack(
        [
            _jax_rollout_trace_patterning(jax.random.PRNGKey(s), N_STEPS)
            for s in range(N_SEEDS)
        ]
    )
    np_us = np_runs[..., 0].mean()
    jax_us = jax_runs[..., 0].mean()
    np_cs = np_runs[..., 1:5].mean()
    jax_cs = jax_runs[..., 1:5].mean()
    np_d = np_runs[..., 5:].mean()
    jax_d = jax_runs[..., 5:].mean()
    np.testing.assert_allclose(jax_us, np_us, atol=TOL)
    np.testing.assert_allclose(jax_cs, np_cs, atol=TOL)
    np.testing.assert_allclose(jax_d, np_d, atol=TOL)


def _np_rollout_noisy_patterning(seed: int, n_steps: int) -> np.ndarray:
    env = NPNoisyPatterning(
        seed=seed,
        ISI_interval=(4, 4),
        ITI_interval=(80, 129),
        gamma=0.9,
        num_CS=4,
        num_activation_patterns=3,
        activation_patterns_prob=0.5,
        num_distractors=0,
        activation_lengths={"CS": 4, "US": 2, "distractor": 4},
        noise=0.1,
    )
    env.reset()
    obs_seq = np.zeros((n_steps + 1, 1 + 4), dtype=np.float32)
    obs_seq[0] = env.observation()
    for t in range(n_steps):
        _, _, _, obs = env.step(None)
        obs_seq[t + 1] = obs
    return obs_seq


def _jax_rollout_noisy_patterning(key, n_steps: int) -> np.ndarray:
    env = animax.NoisyPatterning(
        num_cs=4,
        num_activation_patterns=3,
        activation_patterns_prob=0.5,
        num_distractors=0,
        pattern_seed=0,
    )
    base = env.default_params
    import dataclasses

    params = dataclasses.replace(base, noise=0.1)

    @functools.partial(jax.jit, static_argnames=("n",))
    def go(key, n):
        obs0, state = env.reset(key, params)

        def body(carry, _):
            state, key = carry
            key, k = jax.random.split(key)
            obs, state, _, _, _ = env.step(k, state, 0, params)
            return ((state, key), obs)

        (_, _), seq = jax.lax.scan(body, (state, key), None, length=n)
        return jnp.concatenate([obs0[None], seq], axis=0)

    return np.asarray(go(key, n_steps))


def test_noisy_patterning_mean_activity_matches():
    np_runs = np.stack(
        [_np_rollout_noisy_patterning(s, N_STEPS) for s in range(N_SEEDS)]
    )
    jax_runs = np.stack(
        [
            _jax_rollout_noisy_patterning(jax.random.PRNGKey(s), N_STEPS)
            for s in range(N_SEEDS)
        ]
    )
    np_us = np_runs[..., 0].mean()
    jax_us = jax_runs[..., 0].mean()
    np_cs = np_runs[..., 1:].mean()
    jax_cs = jax_runs[..., 1:].mean()
    np.testing.assert_allclose(jax_us, np_us, atol=TOL)
    np.testing.assert_allclose(jax_cs, np_cs, atol=TOL)
