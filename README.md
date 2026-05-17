# animax

[gymnax](https://github.com/RobertTLange/gymnax) ports of the classical-conditioning
state-construction benchmarks from
[Classical-conditioning-benchmarks-for-state-construction](https://github.com/banafsheh-rafiee/Classical-conditioning-benchmarks-for-state-construction)
(Rafiee et al., *Adaptive Behavior* 2022).

The original repo's `dm_env`-based environments are reimplemented as
`gymnax.environments.Environment` subclasses with immutable PyTree state, so
they `jit` / `vmap` / `scan` cleanly.

## Environments

| Class | Description |
| --- | --- |
| `TraceConditioning` | Single CS, single US, optional Poisson-style distractors. CS at trial start, US after random ISI, ITI between trials. |
| `TracePatterning` | Multi-CS variant: each trial draws a binary CS pattern; the US fires iff the pattern is one of `num_activation_patterns` pre-selected targets (with optional noise flips). Distractors fire at trial start. |
| `NoisyPatterning` | `TracePatterning` with the ISI pinned to the CS activation length (no gap between CS offset and US onset). |

These are *prediction* benchmarks — the action argument is ignored and a
1-element discrete action space is exposed only for gymnax loop compatibility.
The reward / cumulant is the US signal.

## Install

```bash
uv sync                  # core deps
uv sync --extra test     # add pytest + dm-env (for parity tests)
```

## Usage

```python
import jax
import animax

env = animax.TraceConditioning(num_distractors=3)
params = env.default_params

key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)

for _ in range(1000):
    key, k = jax.random.split(key)
    obs, state, reward, done, info = env.step(k, state, 0, params)
```

`step` and `reset` are `jit`-compatible; the env composes with `jax.vmap` and
`jax.lax.scan`.

### Parameters

Each env exposes a `Params` dataclass — override fields with
`dataclasses.replace(env.default_params, ...)`:

```python
import dataclasses
params = dataclasses.replace(env.default_params, isi_min=10, isi_max=10, gamma=0.95)
```

## Tests

```bash
uv run pytest
```

`tests/test_structural.py` checks invariants without relying on the numpy
source. `tests/test_parity_numpy.py` runs the vendored numpy original
side-by-side and compares aggregate firing rates (jax/numpy use different RNGs
so step-by-step equality isn't expected; aggregate statistics agree to within
`atol=0.02`).

## Citation

If you use these benchmarks, cite the original paper:

```bibtex
@article{rafiee2022eye,
  title  = {From Eye-blinks to State Construction: Diagnostic Benchmarks for Online Representation Learning},
  author = {Rafiee, Banafsheh and Abbas, Zaheer and Ghiassian, Sina and Kumaraswamy, Raksha and Sutton, Richard and Ludvig, Elliot and White, Adam},
  journal= {Adaptive Behavior},
  year   = {2022}
}
```
