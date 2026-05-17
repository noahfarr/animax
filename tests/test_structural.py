from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

import animax


def _rollout(env, params, key, n):
    @functools.partial(jax.jit, static_argnames=("n",))
    def go(key, n):
        obs0, state = env.reset(key, params)

        def body(carry, _):
            state, key = carry
            key, k = jax.random.split(key)
            obs, state, r, _, _ = env.step(k, state, 0, params)
            return ((state, key), (obs, r))

        (_, _), (obs_seq, r_seq) = jax.lax.scan(body, (state, key), None, length=n)
        return (obs0, obs_seq, r_seq)

    return go(key, n)


def test_trace_conditioning_cs_fires_at_trial_start():
    env = animax.TraceConditioning(num_distractors=0)
    params = env.default_params
    obs0, _, _ = _rollout(env, params, jax.random.PRNGKey(0), 1)
    assert obs0[0] == 0.0
    assert obs0[1] == 1.0


def test_trace_conditioning_us_follows_cs_within_isi_window():
    env = animax.TraceConditioning(num_distractors=0)
    params = env.default_params
    obs0, obs_seq, r_seq = _rollout(env, params, jax.random.PRNGKey(0), 500)
    full = jnp.concatenate([obs0[None], obs_seq], axis=0)
    us = np.asarray(full[:, 0])
    cs = np.asarray(full[:, 1])
    us_rise = np.where((us[1:] == 1) & (us[:-1] == 0))[0] + 1
    cs_rise = np.concatenate(
        [
            np.array([0]) if cs[0] == 1 else np.array([], dtype=int),
            np.where((cs[1:] == 1) & (cs[:-1] == 0))[0] + 1,
        ]
    )
    assert len(us_rise) >= 1, "expected at least one US firing in 500 steps"
    for u in us_rise:
        preceding = cs_rise[cs_rise <= u]
        assert preceding.size > 0
        delta = int(u - preceding[-1])
        assert params.isi_min <= delta <= params.isi_max


def test_trace_conditioning_activation_lengths():
    env = animax.TraceConditioning(num_distractors=0)
    params = env.default_params
    obs0, obs_seq, _ = _rollout(env, params, jax.random.PRNGKey(1), 500)
    full = jnp.concatenate([obs0[None], obs_seq], axis=0)
    us = np.asarray(full[:, 0])
    cs = np.asarray(full[:, 1])

    def run_lengths(x: np.ndarray) -> list[int]:
        runs = []
        n = 0
        for v in x:
            if v == 1:
                n += 1
            elif n:
                runs.append(n)
                n = 0
        if n:
            runs.append(n)
        return runs

    cs_runs = run_lengths(cs)
    us_runs = run_lengths(us)
    assert all((r == params.cs_activation_length for r in cs_runs[:-1]))
    assert all((r == params.us_activation_length for r in us_runs[:-1]))


def test_jit_and_vmap_and_scan_compile():
    env = animax.TracePatterning(num_cs=4, num_activation_patterns=3, num_distractors=2)
    params = env.default_params

    @functools.partial(jax.jit, static_argnames=("n",))
    def go(key, n):
        obs0, state = env.reset(key, params)

        def body(carry, _):
            state, key = carry
            key, k = jax.random.split(key)
            obs, state, r, _, _ = env.step(k, state, 0, params)
            return ((state, key), (obs, r))

        (_, _), (obs_seq, r_seq) = jax.lax.scan(body, (state, key), None, length=n)
        return (obs_seq, r_seq)

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    batched = jax.vmap(lambda k: go(k, 200))
    obs_b, r_b = batched(keys)
    assert obs_b.shape == (4, 200, 1 + 4 + 2)
    assert r_b.shape == (4, 200)


def test_noisy_patterning_pins_isi():
    env = animax.NoisyPatterning(num_cs=4, num_activation_patterns=3)
    params = env.default_params
    assert params.isi_min == params.cs_activation_length
    assert params.isi_max == params.cs_activation_length


def test_trace_patterning_pattern_matches_drive_us():
    env = animax.TracePatterning(
        num_cs=2,
        num_activation_patterns=2,
        activation_patterns_prob=0.0,
        num_distractors=0,
        pattern_seed=0,
    )
    params = env.default_params
    _, _, r_seq = _rollout(env, params, jax.random.PRNGKey(2), 1500)
    r = np.asarray(r_seq)
    assert r.sum() > 0
