import jax.numpy as jnp
from jax import Array

NO_ONSET = jnp.iinfo(jnp.int32).min // 2


def is_active(time_step: Array, onset: Array, activation_length: int) -> Array:
    return (time_step >= onset) & (time_step < onset + activation_length)
