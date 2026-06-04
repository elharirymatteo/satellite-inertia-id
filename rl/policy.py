"""Tiny MLP policy for the T1 env.

Returns torque commands in [-tau_max, tau_max] given an observation. Pure JAX
functions; parameter dict is passed in explicitly (no Flax/Haiku to keep deps
minimal).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def init_params(key, obs_dim: int, hidden: int = 64, action_dim: int = 3):
    """Two-hidden-layer MLP with tanh hidden + tanh output."""
    k1, k2, k3 = jax.random.split(key, 3)
    scale = jnp.sqrt(2.0)
    W1 = scale * jax.random.normal(k1, (obs_dim, hidden)) / jnp.sqrt(obs_dim)
    b1 = jnp.zeros(hidden)
    W2 = scale * jax.random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden)
    b2 = jnp.zeros(hidden)
    # Small final weights so the policy starts near zero (low torque, safe).
    W3 = 0.01 * jax.random.normal(k3, (hidden, action_dim)) / jnp.sqrt(hidden)
    b3 = jnp.zeros(action_dim)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def apply(params, obs: jnp.ndarray, tau_max: float):
    h = jnp.tanh(obs @ params["W1"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    raw = h @ params["W3"] + params["b3"]
    return tau_max * jnp.tanh(raw)
