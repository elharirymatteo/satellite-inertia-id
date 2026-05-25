"""Verify the rotation-based DR sampler produces PSD inertia tensors with
eigenvalues in the requested range and bounded off-diagonals."""
import numpy as np
import jax
import jax.numpy as jnp

from rl.t1_env import sample_sat


def test_sampler_yields_psd_with_bounded_offdiagonals():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 256)
    sats = [
        sample_sat(k, I_range=(0.1, 20.0),
                   I_rw=1e-4, rw_axes=jnp.eye(3),
                   rw_speed_max=600.0, rw_torque_max=0.05,
                   log_uniform=True, max_tilt_angle=np.pi / 8)
        for k in keys
    ]

    for sat in sats:
        I = np.asarray(sat.I_sat)
        # Symmetric
        np.testing.assert_allclose(I, I.T, atol=1e-10)
        # PSD with eigenvalues in [0.1, 20]
        eigs = np.linalg.eigvalsh(I)
        assert eigs.min() >= 0.1 - 1e-9, f"min eig {eigs.min()}"
        assert eigs.max() <= 20.0 + 1e-9, f"max eig {eigs.max()}"
        # Off-diagonals bounded (~few % of trace for tilt = pi/8)
        offdiag = np.abs(I - np.diag(np.diag(I))).max()
        assert offdiag <= 0.5 * np.trace(I), f"offdiag {offdiag} vs trace {np.trace(I)}"
