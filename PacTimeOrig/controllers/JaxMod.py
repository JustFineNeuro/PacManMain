import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import heapq
from matplotlib.animation import FuncAnimation
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import minimize, differential_evolution,NonlinearConstraint
import multiprocessing
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
import optax



#TODO loss for inner (lbfgs/adam) _p



# Define the Loss Functions

def create_loss_function_inner(generate_rbf_basis, num_rbfs, generate_smoothing_penalty, lambda_reg, ctrltype='pv',
                               opttype='first'):
    @jit
    def loss_function(params, inputs):
        """
        Computes the negative log-likelihood.

        params: Tuple containing (weights, widths, L1_flat, L2_flat)
        inputs: Dictionary containing necessary inputs
        """

        # Converting adam and lbfgs to generic loss call:
        gainsize = {
            'p': 1,
            'pv': 2, 'pf': 2,
            'pvi': 3, 'pif': 3, 'pvf': 3,
            'pvif': 4,
        }.get(ctrltype, 1)

        # Split the flat parameter vector into components
        weights = params[:num_rbfs]  # Shape: (num_rbfs, )
        widths = params[num_rbfs]

        # Get gain parameters
        L1 = params[num_rbfs + 1:num_rbfs + (gainsize + 1)]
        L2 = params[(num_rbfs + (gainsize + 1)):num_rbfs + (2 * gainsize + 1)]
        if opttype == 'first':
            # Apply Softplus to ensure positivity with 1st order gradient optimizer
            L1 = jnp.log(1 + jnp.exp(L1))  # Softplus transformation
            L2 = jnp.log(1 + jnp.exp(L2))  # Softplus transformation
        elif opttype == 'second':
            pass

        # weights, widths, L1, L2 = params
        x0 = inputs['x0']  # Initial state (position and velocity)
        SetpointA_pos = inputs['SetpointA_pos']
        SetpointA_vel = inputs['SetpointA_vel']
        SetpointA_accel = inputs['SetpointA_accel']
        SetpointB_pos = inputs['SetpointB_pos']
        SetpointB_vel = inputs['SetpointB_vel']
        SetpointB_accel = inputs['SetpointB_accel']

        u_obs = inputs['u_obs']
        tmp = inputs['tmp']
        centers = inputs['centers']
        A = inputs['A']
        B = inputs['B']
        dt = inputs['dt']

        N = SetpointA_pos.shape[0]

        # Generate RBF basis functions using precomputed centers
        X = generate_rbf_basis(tmp, centers, widths)
        tmpkernel = jnp.dot(X, weights)
        w1 = jax.nn.sigmoid(tmpkernel)
        w2 = 1 - w1

        # Initialize state and control outputs
        x = jnp.zeros((N + 1, A.shape[1]))
        x = x.at[0].set(x0)
        u_out = jnp.zeros((N, B.shape[1]))
        # Initialize integrator variables
        int_e_pos_1 = jnp.zeros(2)
        int_e_pos_2 = jnp.zeros(2)
        if ctrltype == 'p':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                e1 = jnp.vstack((e_pos_1))
                e2 = jnp.vstack((e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 * e1
                u2 = -L2 * e2
                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + (B @ u).flatten()
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u.flatten())
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        if ctrltype == 'pv':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e1 = jnp.vstack((e_pos_1, e_vel_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2

                # Update state
                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))

        # Compute negative log-likelihood
        residuals = u_out - u_obs
        l = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)
        S_x = generate_smoothing_penalty(num_rbfs)
        regularization = lambda_reg * (weights @ S_x @ weights.transpose())
        loss = -l + regularization
        return loss

    return loss_function


def create_loss_function_inner_slack(generate_rbf_basis, num_rbfs, generate_smoothing_penalty, lambda_reg,
                                     ctrltype='pv',
                                     opttype='first'):
    @jit
    def loss_function(params, inputs):
        """
        Computes the negative log-likelihood.

        params: Tuple containing (weights, widths, L1_flat, L2_flat)
        inputs: Dictionary containing necessary inputs
        """

        # Converting adam and lbfgs to generic loss call:
        gainsize = {
            'p': 1,
            'pv': 2, 'pf': 2,
            'pvi': 3, 'pif': 3, 'pvf': 3,
            'pvif': 4,
        }.get(ctrltype, 1)

        # Split the flat parameter vector into components
        weights = params[:(2 * num_rbfs)]  # Shape: (num_rbfs, )
        weights_1 = weights[0:num_rbfs]
        weights_2 = weights[num_rbfs:]
        widths = params[2 * num_rbfs]

        # Get gain parameters

        L1 = params[(2 * num_rbfs) + 1:(2 * num_rbfs) + (gainsize + 1)]
        L2 = params[((2 * num_rbfs) + (gainsize + 1)):((2 * num_rbfs) + (2 * gainsize)) + 1]

        # Get slack parameter:
        alpha = params[-1]

        if opttype == 'first':
            # Apply Softplus to ensure positivity with 1st order gradient optimizer
            L1 = jnp.log(1 + jnp.exp(L1))  # Softplus transformation
            L2 = jnp.log(1 + jnp.exp(L2))  # Softplus transformation
        elif opttype == 'second':
            pass

        # weights, widths, L1, L2 = params
        x0 = inputs['x0']  # Initial state (position and velocity)
        SetpointA_pos = inputs['SetpointA_pos']
        SetpointA_vel = inputs['SetpointA_vel']
        SetpointA_accel = inputs['SetpointA_accel']
        SetpointB_pos = inputs['SetpointB_pos']
        SetpointB_vel = inputs['SetpointB_vel']
        SetpointB_accel = inputs['SetpointB_accel']

        u_obs = inputs['u_obs']
        tmp = inputs['tmp']
        centers = inputs['centers']
        A = inputs['A']
        B = inputs['B']
        dt = inputs['dt']

        N = SetpointA_pos.shape[0]

        # Hidden wegiht softmax
        # Generate RBF basis functions using precomputed centers
        X = generate_rbf_basis(tmp, centers, widths)
        z1 = jnp.dot(X, weights_1)
        z2 = jnp.dot(X, weights_2)

        # Manual softmax transformation

        w1 = jnp.exp(z1) / (jnp.exp(z1) + jnp.exp(z2) + 1.0)
        w2 = jnp.exp(z2) / (jnp.exp(z1) + jnp.exp(z2) + 1.0)
        w3 = 1.0 / (jnp.exp(z1) + jnp.exp(z2) + 1.0)

        # Initialize state and control outputs
        x = jnp.zeros((N + 1, A.shape[1]))
        x = x.at[0].set(x0)
        u_out = jnp.zeros((N, B.shape[1]))
        # Initialize integrator variables
        int_e_pos_1 = jnp.zeros(2)
        int_e_pos_2 = jnp.zeros(2)
        if ctrltype == 'p':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                e1 = jnp.vstack((e_pos_1))
                e2 = jnp.vstack((e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 * e1
                u2 = -L2 * e2
                u3 = jnp.array(-alpha * x[k, 2:].reshape(-1, 1))
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + (B @ u).flatten()
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u.flatten())
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        if ctrltype == 'pv':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e1 = jnp.vstack((e_pos_1, e_vel_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2))

                # Compute control inputs using the estimated gains
                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                # Update state
                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))

        # Compute negative log-likelihood
        residuals = u_out - u_obs
        l = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)
        S_x = generate_smoothing_penalty(num_rbfs)
        regularization = lambda_reg * (weights_1 @ S_x @ weights_1.transpose())
        regularizationb = lambda_reg * (weights_2 @ S_x @ weights_2.transpose())
        loss = -l + (regularization + regularizationb)*0.5

        return loss

    return loss_function


## Optimization functions: Trust/Lbfgs


def stability_constraints(params_flat, inputs, gainsize, multip, epsilon=1e-3):
    """
    Computes stability constraints for both controllers.

    Parameters:
    - params_flat: 1D array of all parameters being optimized.
    - inputs: Dictionary containing necessary inputs.
    - gainsize: Number of gains per controller based on ctrltype.
    - multip: Multiplier based on slack_model (1 or 2).
    - epsilon: Small buffer to ensure eigenvalues are strictly inside the unit circle.

    Returns:
    - constraints: 1D array containing constraint values for both controllers.
                   Each value should be >= 0 to satisfy 1 - max_eig - epsilon >= 0.
    """
    # Extract positions based on slack_model
    if inputs.get('slack_model', False):
        # slack_model is True
        weights = params_flat[:inputs['num_rbfs']]  # Shape: (num_rbfs, )
        widths = params_flat[inputs['num_rbfs']]
        K1 = params_flat[inputs['num_rbfs'] + 1:inputs['num_rbfs'] + (gainsize + 1)]
        K2 = params_flat[(inputs['num_rbfs'] + (gainsize + 1)):(inputs['num_rbfs'] + (gainsize * 2 + 1))]
        # If alpha exists, it's at the end; ignore for gains extraction
    else:
        # slack_model is False
        weights = params_flat[:inputs['num_rbfs']]  # Shape: (num_rbfs, )
        widths = params_flat[inputs['num_rbfs']]
        K1 = params_flat[inputs['num_rbfs'] + 1:inputs['num_rbfs'] + (gainsize + 1)]
        K2 = params_flat[(inputs['num_rbfs'] + (gainsize + 1)):(inputs['num_rbfs'] + (gainsize * 2 + 1))]

    # Define system matrices A and B
    # These should be part of inputs; adjust accordingly
    # Assuming 'A' and 'B' are provided in inputs
    A = inputs['A']
    B = inputs['B']

    #Expand gain matrices:
    K1_expanded = np.tile(K1, (2, 1))  # Repeat along the second dimension
    K2_expanded = np.tile(K2, (2, 1))  # Repeat along the second dimension


    # Compute closed-loop A matrices for both controllers
    #A-BK
    A_cl1 = A - B @ K1_expanded  # Shape: same as A
    A_cl2 = A - B @ K2_expanded

    # Compute eigenvalues
    eigvals1 = np.linalg.eigvals(A_cl1)
    eigvals2 = np.linalg.eigvals(A_cl2)

    # Compute maximum eigenvalue magnitudes
    max_eig1 = np.max(np.abs(eigvals1))
    max_eig2 = np.max(np.abs(eigvals2))

    # Compute constraint values: 1 - max_eig - epsilon >= 0
    constraint1 = 1 - max_eig1 - epsilon
    constraint2 = 1 - max_eig2 - epsilon

    return np.array([constraint1, constraint2])

def outer_optimization_lbfgs(inputs, loss_function, grad_loss,hessian_loss=None, ctrltype='pvi',randomize_weights=True,maxiter=10000,tolerance=1e-6,optimizer='trust',slack_model=False,stable_constraint=False):
    """
    Performs joint optimization of L1, L2, weights, and widths.
    Uses a system stability contraint to on the controller gains:
    explanation: A_aug=A-BK. The eigenvalues
    inputs: Dictionary containing necessary inputs.
    loss_function: JAX-compiled loss function.
    grad_loss: JAX-compiled gradient of the loss function.
    """
    # Total parameters:
    # Converting adam and lbfgs to generic loss call:
    gainsize = {
        'p': 1,
        'pv': 2, 'pf': 2,
        'pvi': 3, 'pif': 3, 'pvf': 3,
        'pvif': 4,
    }.get(ctrltype, 1)


    # Initial guess
    if randomize_weights is True:
        setit=1
    else:
        setit=0

    if slack_model is False:
        multip = 1
    elif slack_model is True:
        multip = 2


    if slack_model is False:
        init_weights = np.zeros(inputs['num_rbfs'])+setit*np.ones_like(np.zeros(inputs['num_rbfs']))*np.random.randn(inputs['num_rbfs'])
        init_widths = np.ones(1)*2.0
        init_gains = 2.0*(np.abs(np.random.random(gainsize*2))).flatten()
        initial_guess = np.concatenate((init_weights,init_widths,init_gains))
    elif slack_model is True:
        init_weights = np.zeros(inputs['num_rbfs']*2)+setit*np.ones_like(np.zeros(2*inputs['num_rbfs']))*np.random.randn(2*inputs['num_rbfs'])
        init_widths = np.ones(1)*2.0
        init_gains = 2.0*(np.abs(np.random.random(gainsize*2))).flatten()
        init_alpha = (np.abs(np.random.normal(1))*3).flatten()
        initial_guess = np.concatenate((init_weights,init_widths,init_gains,init_alpha))


    # Define bounds for optimizer
    lower_weight_bound = -40.0
    upper_weight_bound = 40.0
    width_lower_bound = 0.001
    width_upper_bound = 15.0
    gain_lower_bound = 0.01
    gain_upper_bound = 40.0
    alpha_lower_bound = 0.00001
    alpha_upper_bound = 30.0


    weight_bounds = [(lower_weight_bound, upper_weight_bound)] * (inputs['num_rbfs']*multip)
    width_bounds = [(width_lower_bound, width_upper_bound)]
    gain_bounds = [(gain_lower_bound, gain_upper_bound)] * gainsize*2
    alpha_bounds = [(alpha_lower_bound, alpha_upper_bound)]
    if slack_model is False:
        bounds = weight_bounds + width_bounds + gain_bounds
    elif slack_model is True:
        bounds = weight_bounds + width_bounds + gain_bounds + alpha_bounds



    # Define the objective function
    def objective(params_flat):
        return float(loss_function(params_flat, inputs))

    # Define the gradient function
    def optimizer_gradient(params_flat):
        grads = grad_loss(params_flat, inputs)
        grads_flat = np.array(grads)
        return grads_flat

    # Define the Hessian function (optional)
    def optimizer_hessian(params_flat):
        if hessian_loss is not None:
            hess = hessian_loss(params_flat, inputs)
            return np.array(hess)
        else:
            raise ValueError("Hessian function was not provided.")

    if optimizer == 'trust':
        # Run the optimizer
        result = minimize(
            objective,
            initial_guess,
            method='trust-constr',
            jac=optimizer_gradient,
            hess=optimizer_hessian if hessian_loss else None,  # Include if available
            bounds=bounds,
            tol=tolerance,
            options={
                'gtol': 1e-15,  # Tolerance for the gradient norm
                'xtol': 1e-20,  # Tolerance for the change in solution
                'barrier_tol': 1e-6,  # Tolerance for the barrier parameter
                'maxiter': maxiter,  # Maximum number of iterations
                'disp': True  # Verbosity level (optional, useful for debugging)
            }
        )

    elif optimizer == 'lbfgs':
        # Run the optimizer
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            jac=optimizer_gradient,
            hess=optimizer_hessian if hessian_loss else None,  # Include if available
            bounds=bounds,
            tol=tolerance,
            options={'maxiter': maxiter, 'disp': True, 'ftol': 1e-15, 'gtol': 1e-10,'maxfun': 10000},
        )
        # options = {'maxiter': maxiter, 'disp': True, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 10000},

    best_params_flat = result.x
    best_loss = result.fun

    # Gather paramters and put in tuple
    if slack_model is False:
        weights = best_params_flat[:inputs['num_rbfs']]  # Shape: (num_rbfs, )
        widths = best_params_flat[inputs['num_rbfs']]
        L1 = best_params_flat[inputs['num_rbfs'] + 1:inputs['num_rbfs'] + (gainsize + 1)]
        L2 = best_params_flat[(inputs['num_rbfs'] + (gainsize + 1)):(inputs['num_rbfs'] + (gainsize * 2 + 1))]
        outtuple=(L1,L2,weights,widths)

    elif slack_model is True:
        weights = best_params_flat[:(multip*inputs['num_rbfs'])]  # Shape: (num_rbfs, )
        widths = best_params_flat[multip*inputs['num_rbfs']]
        L1 = best_params_flat[(multip*inputs['num_rbfs'] + 1):(multip*inputs['num_rbfs'] + (gainsize + 1))]
        L2 = best_params_flat[((multip*inputs['num_rbfs']) + (gainsize + 1)):((multip*inputs['num_rbfs']) + (gainsize * 2 + 1))]
        alpha = best_params_flat[-1]
        outtuple=(L1,L2,weights,widths,alpha)


    return outtuple,best_params_flat, best_loss




## Optimization functions: first-order

def initialize_parameters(inputs, ctrltype='pvi',randomize_weights=True, slack_model=False):

    gainsize = {
        'p': 1,
        'pv': 2, 'pf': 2,
        'pvi': 3, 'pif': 3, 'pvf': 3,
        'pvif': 4,
    }.get(ctrltype, 1)

    key = jax.random.PRNGKey(1)  # Seed for reproducibility

    widths = jnp.array(2.0)
    # Initialize L1 and L2 gains
    key, subkey = jax.random.split(key)
    L1 = jnp.exp(jax.random.normal(subkey, shape=(gainsize, 1)))  # Shape: (2, 4)

    key, subkey = jax.random.split(key)
    L2 = jnp.exp(jax.random.normal(subkey, shape=(gainsize, 1)))  # Shape: (2, 4)

    # Flatten L1 and L2 for optimization
    L1_flat = L1.flatten()  # Shape: (8, )
    L2_flat = L2.flatten()  # Shape: (8, )

    # Initial guess
    if randomize_weights is True:
        setit = 1
    else:
        setit = 0
    if slack_model is False:
        weights = (jnp.zeros(inputs['num_rbfs'])) + setit * np.ones_like(np.zeros(inputs['num_rbfs'])) * np.random.randn(inputs['num_rbfs'])
        #    Combine all parameters into a single tuple
        params = (weights, widths, L1_flat, L2_flat)
    elif slack_model is True:
        weights = (jnp.zeros(inputs['num_rbfs'] * 2)) + setit * np.ones_like(np.zeros(inputs['num_rbfs'])) * np.random.randn(inputs['num_rbfs'])
        log_alpha = jnp.exp(jax.random.normal(subkey, shape=(1)))
        params = (weights, widths, L1_flat, L2_flat, log_alpha)

    return params


def setup_optimizer(params, optimizer='adam', learning_rate=1e-3, slack_model=False):
    # Flatten all parameters into a single vector for optimization
    if slack_model == False:
        params_flat = jnp.concatenate([
            params[0].flatten(),  # weights
            params[1].flatten(),  # widths
            params[2],  # L1_flat
            params[3],  # L2_flat
        ])
    elif slack_model == True:
        params_flat = jnp.concatenate([
            params[0].flatten(),  # weights
            params[1].flatten(),  # widths
            params[2],  # L1_flat
            params[3],  # L2_flat
            params[4]  #log alpha
        ])
    if optimizer == 'adam':
        # Define the optimizer (Adam)
        optimizer = optax.adam(learning_rate)
    elif optimizer == 'amsgrad':
        optimizer = optax.amsgrad(learning_rate)

    # Initialize optimizer state
    opt_state = optimizer.init(params_flat)

    return optimizer, opt_state


def optimization_step(params, opt_state, optimizer, loss_function, inputs, ctrltype='p',slack_model=True):
    gainsize = {
        'p': 1,
        'pv': 2, 'pf': 2,
        'pvi': 3, 'pif': 3, 'pvf': 3,
        'pvif': 4,
    }.get(ctrltype, 1)

    # Flatten parameters
    if slack_model == False:
        params_flat = jnp.concatenate([
            params[0].flatten(),  # weights
            params[1].flatten(),  # widths
            params[2],  # L1_flat
            params[3],  # L2_flat
        ])
    elif slack_model == True:
        params_flat = jnp.concatenate([
            params[0].flatten(),  # weights
            params[1].flatten(),  # widths
            params[2].flatten(),  # L1_flat
            params[3].flatten(),  # L2_flat
            params[4].flatten(),  # log alpha
        ])

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_function)(params_flat, inputs)

    # Update parameters using the optimizer
    updates, opt_state = optimizer.update(grads, opt_state)
    params_flat = optax.apply_updates(params_flat, updates)

    # Unflatten parameters back to original shapes
    num_weights = params[0].shape[0]
    weights = params_flat[:num_weights]
    widths = params_flat[num_weights]
    if slack_model == False:
        llflat  = params_flat[(1 + num_weights):]
        L1_flat = llflat[0:gainsize]
        L2_flat = llflat[gainsize:]
        # Return updated parameters, optimizer state, and loss
        new_params = (weights, widths, L1_flat, L2_flat)

    elif slack_model is True:
        llflat=params_flat[(1 + num_weights):]
        #Leaves just L and alpha
        #grab alpha from end
        log_alpha = llflat[-1]

        L1_flat = llflat[0:gainsize]
        L2_flat = llflat[gainsize:-1]

        # Return updated parameters, optimizer state, and loss
        new_params = (weights, widths, L1_flat, L2_flat,log_alpha)

    return new_params, opt_state, loss

