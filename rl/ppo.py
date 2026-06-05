"""Minimal PPO in pure JAX for the F5 ablation.

Gaussian policy (state-independent log_std), separate value head, GAE,
clipped surrogate loss. Designed as the model-free baseline against the
diff-sim path-derivative policy gradient used by t1_train_dr_ekf.py —
shows whether the wins of the diff-sim approach come from differentiable
simulation (which PPO doesn't exploit) or from the env/reward design
alone (which PPO can also exploit, just with more samples).

Reuses rl.policy for the actor mean network so the architecture matches
the diff-sim policy exactly.
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax

from rl import policy as policy_mod


def init_ppo_params(key, obs_dim: int, action_dim: int = 3, hidden: int = 64):
    k_actor, k_critic, k_logstd = jax.random.split(key, 3)
    actor = policy_mod.init_params(k_actor, obs_dim, hidden, action_dim)
    critic = policy_mod.init_params(k_critic, obs_dim, hidden, action_dim=1)
    log_std = -0.5 * jnp.ones((action_dim,))   # std ~= 0.6
    return {"actor": actor, "critic": critic, "log_std": log_std}


def actor_mean(params, obs):
    h = jnp.tanh(obs @ params["actor"]["W1"] + params["actor"]["b1"])
    h = jnp.tanh(h @ params["actor"]["W2"] + params["actor"]["b2"])
    return h @ params["actor"]["W3"] + params["actor"]["b3"]


def critic_value(params, obs):
    h = jnp.tanh(obs @ params["critic"]["W1"] + params["critic"]["b1"])
    h = jnp.tanh(h @ params["critic"]["W2"] + params["critic"]["b2"])
    return (h @ params["critic"]["W3"] + params["critic"]["b3"])[..., 0]


def sample_action(params, obs, key, tau_max: float):
    """Return (action, log_prob). Action is clipped to [-tau_max, tau_max]."""
    mean = actor_mean(params, obs)
    std = jnp.exp(params["log_std"])
    noise = jax.random.normal(key, mean.shape)
    raw = mean + std * noise
    action = jnp.clip(raw, -tau_max, tau_max)
    log_prob = (-0.5 * ((raw - mean) / std) ** 2
                - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)).sum(axis=-1)
    return action, log_prob


def log_prob(params, obs, action):
    """Log-prob of action under current policy. Used in PPO ratio."""
    mean = actor_mean(params, obs)
    std = jnp.exp(params["log_std"])
    return (-0.5 * ((action - mean) / std) ** 2
            - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)).sum(axis=-1)


def compute_gae(rewards, values, last_value, gamma: float, lam: float):
    """Generalized Advantage Estimation. Inputs:
      rewards:    (T,)
      values:     (T,)   value at each visited state
      last_value: scalar value at state T (bootstrap)
    Returns (advantages (T,), returns (T,))."""
    def body(carry, inp):
        gae = carry
        r, v, v_next = inp
        delta = r + gamma * v_next - v
        gae = delta + gamma * lam * gae
        return gae, gae

    values_next = jnp.concatenate([values[1:], jnp.array([last_value])])
    _, advantages = jax.lax.scan(
        body, jnp.array(0.0), (rewards, values, values_next), reverse=True,
    )
    returns = advantages + values
    return advantages, returns


def ppo_loss(params, batch, clip_eps: float, value_coef: float, ent_coef: float):
    obs, actions, old_log_probs, advantages, returns = batch
    new_log_probs = log_prob(params, obs, actions)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.minimum(surr1, surr2).mean()

    values = critic_value(params, obs)
    value_loss = ((returns - values) ** 2).mean()

    std = jnp.exp(params["log_std"])
    entropy = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * std ** 2).sum()

    return policy_loss + value_coef * value_loss - ent_coef * entropy
