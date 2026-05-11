"""
Observability metrics for spacecraft inertia identification.

Quantifies how well a given torque profile excites the system dynamics
for the purpose of inertia parameter estimation.
"""

import numpy as np


def compute_observability_metric(torques, dt, I_ref, normalize_energy=True):
    """
    Compute observability metric for a torque profile.

    Simulates simplified rigid-body dynamics (Euler-forward, diagonal inertia,
    no reaction wheels), builds the linear regressor W for the least-squares
    problem, and derives SVD-based observability quantities.

    Parameters
    ----------
    torques : ndarray, shape (N, 3)
        Applied torques [Nm] for each time step.
    dt : float
        Time step [s].
    I_ref : tuple of float
        Reference diagonal inertia (Ixx, Iyy, Izz) [kg·m²].
    normalize_energy : bool
        If True, divide W by sqrt(sum(torques**2) * dt) + 1e-12.

    Returns
    -------
    dict with keys:
        min_sv          : float – smallest singular value of W
        max_sv          : float – largest singular value of W
        condition_number: float – max_sv / (min_sv + 1e-12)
        log_det         : float – log-determinant of Fisher info F = W^T W
        score           : float – composite observability score
    """
    N = torques.shape[0]
    Ixx, Iyy, Izz = float(I_ref[0]), float(I_ref[1]), float(I_ref[2])

    # Simulate Euler-forward rigid-body dynamics
    omega = np.zeros(3)
    omega_hist = np.zeros((N, 3))
    domega_hist = np.zeros((N, 3))

    for k in range(N):
        tau = torques[k]
        wx, wy, wz = omega

        # Euler's equations: I dω/dt = τ − ω × (I ω)
        domega = np.array([
            (tau[0] - (Izz - Iyy) * wy * wz) / Ixx,
            (tau[1] - (Ixx - Izz) * wx * wz) / Iyy,
            (tau[2] - (Iyy - Ixx) * wx * wy) / Izz,
        ])

        omega_hist[k] = omega
        domega_hist[k] = domega
        omega = omega + dt * domega

    # Build regressor W (shape 3N x 3)
    # For diagonal I = diag(a, b, c), Euler's equations linearised in (a, b, c):
    #   tau_x = a*dw_x  + (-wy*wz)*b + (wy*wz)*c   → row = [dw_x,  -wy*wz,  wy*wz]
    #   tau_y = (wx*wz)*a + b*dw_y  + (-wx*wz)*c    → row = [wx*wz,  dw_y,  -wx*wz]
    #   tau_z = (-wx*wy)*a + (wx*wy)*b + c*dw_z      → row = [-wx*wy, wx*wy,  dw_z ]
    wx = omega_hist[:, 0]
    wy = omega_hist[:, 1]
    wz = omega_hist[:, 2]
    dw_x = domega_hist[:, 0]
    dw_y = domega_hist[:, 1]
    dw_z = domega_hist[:, 2]

    row_x = np.column_stack([dw_x,   -wy * wz,  wy * wz])
    row_y = np.column_stack([wx * wz,  dw_y,    -wx * wz])
    row_z = np.column_stack([-wx * wy, wx * wy,  dw_z])

    # Interleave: (row_x[0], row_y[0], row_z[0], row_x[1], ...)
    W = np.empty((3 * N, 3))
    W[0::3] = row_x
    W[1::3] = row_y
    W[2::3] = row_z

    if normalize_energy:
        energy = np.sqrt(np.sum(torques ** 2) * dt) + 1e-12
        W = W / energy

    # SVD-based metrics
    sv = np.linalg.svd(W, compute_uv=False)
    min_sv = float(sv[-1])
    max_sv = float(sv[0])
    condition_number = max_sv / (min_sv + 1e-12)

    # log-det of Fisher info F = W^T W  (= 2 * sum(log(sv)))
    log_det = float(2.0 * np.sum(np.log(sv + 1e-300)))

    # Composite score: reward large min singular value, penalise poor conditioning
    score = min_sv / (np.log10(condition_number + 1) + 1e-12)

    return {
        'min_sv': min_sv,
        'max_sv': max_sv,
        'condition_number': condition_number,
        'log_det': log_det,
        'score': score,
    }


def compute_fim_from_data(omega, domega):
    """
    Compute the Fisher Information Matrix for diagonal inertia estimation
    from actual observed angular velocity and acceleration data.

    Builds the regressor W from Euler's equations linearised in (Ixx, Iyy, Izz):
        tau_x ≈ Ixx*dω_x − Iyy*ω_y*ω_z + Izz*ω_y*ω_z   → row = [dω_x, −ω_y*ω_z,  ω_y*ω_z]
        tau_y ≈ Ixx*ω_z*ω_x + Iyy*dω_y  − Izz*ω_z*ω_x   → row = [ ω_z*ω_x,  dω_y, −ω_z*ω_x]
        tau_z ≈ −Ixx*ω_x*ω_y + Iyy*ω_x*ω_y + Izz*dω_z   → row = [−ω_x*ω_y, ω_x*ω_y,  dω_z]

    F = W^T W is the Fisher Information Matrix; its eigenvalues set the
    Cramér-Rao lower bound on estimation variance.

    Parameters
    ----------
    omega  : ndarray, shape (N, 3) — observed angular velocity [rad/s]
    domega : ndarray, shape (N, 3) — observed angular acceleration [rad/s²]

    Returns
    -------
    dict with keys:
        W              : ndarray (3N, 3) — regressor matrix
        F              : ndarray (3, 3)  — Fisher information matrix W^T W
        log_det_F      : float           — log-determinant of F (D-optimality)
        min_sv         : float           — smallest singular value of W
        condition_number : float         — max_sv / (min_sv + 1e-300)
    """
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    dw_x, dw_y, dw_z = domega[:, 0], domega[:, 1], domega[:, 2]

    row_x = np.column_stack([ dw_x,    -wy * wz,  wy * wz])
    row_y = np.column_stack([ wz * wx,  dw_y,    -wz * wx])
    row_z = np.column_stack([-wx * wy,  wx * wy,  dw_z])

    N = omega.shape[0]
    W = np.empty((3 * N, 3))
    W[0::3] = row_x
    W[1::3] = row_y
    W[2::3] = row_z

    F = W.T @ W
    sv = np.linalg.svd(W, compute_uv=False)
    min_sv = float(sv[-1])
    max_sv = float(sv[0])

    return {
        'W': W,
        'F': F,
        'log_det_F': float(2.0 * np.sum(np.log(sv + 1e-300))),
        'min_sv': min_sv,
        'condition_number': max_sv / (min_sv + 1e-300),
    }


def score_profiles_canonical(torques_dict, dt, I_ref, normalize_energy=True):
    """
    Score multiple torque profiles and rank them by observability.

    Parameters
    ----------
    torques_dict : dict[str, ndarray]
        Mapping from profile name to torque array (N, 3).
    dt : float
        Time step [s].
    I_ref : tuple of float
        Reference diagonal inertia (Ixx, Iyy, Izz).
    normalize_energy : bool
        Passed through to compute_observability_metric.

    Returns
    -------
    dict[str, dict]
        Same keys as torques_dict; each value is the metric dict from
        compute_observability_metric, with an added 'rank' key (1 = best).
    """
    results = {
        name: compute_observability_metric(torques, dt=dt, I_ref=I_ref,
                                           normalize_energy=normalize_energy)
        for name, torques in torques_dict.items()
    }

    # Sort descending by score; assign rank 1 to highest scorer
    ranked = sorted(results.items(), key=lambda kv: kv[1]['score'], reverse=True)
    for rank, (name, _) in enumerate(ranked, start=1):
        results[name]['rank'] = rank

    return results
