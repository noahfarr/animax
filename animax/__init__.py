from animax.noisy_patterning import NoisyPatterning
from animax.trace_conditioning import (
    TraceConditioning,
    TraceConditioningParams,
    TraceConditioningState,
)
from animax.trace_patterning import (
    TracePatterning,
    TracePatterningParams,
    TracePatterningState,
)

REGISTRY = {
    "TraceConditioning-v0": TraceConditioning,
    "TracePatterning-v0": TracePatterning,
    "NoisyPatterning-v0": NoisyPatterning,
}


def make(env_id: str, **env_kwargs):
    if env_id not in REGISTRY:
        raise ValueError(f"Unknown env_id {env_id!r}. Registered: {sorted(REGISTRY)}")
    env = REGISTRY[env_id](**env_kwargs)
    return env, env.default_params


__all__ = [
    "REGISTRY",
    "NoisyPatterning",
    "TraceConditioning",
    "TraceConditioningParams",
    "TraceConditioningState",
    "TracePatterning",
    "TracePatterningParams",
    "TracePatterningState",
    "make",
]
