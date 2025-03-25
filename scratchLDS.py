import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.optimize import root
from PacTimeOrig.models import NNtorch
from PacTimeOrig.models.utils import find_fixed_points_with_nullspace
from sklearn.utils.extmath import randomized_svd




#TODO: notes we should be passing to the RNN for line attractors the
# utilde we derived for adversarial. See if it can distinguish then.
#TODO: finite time lyap



#
def generate_emissions(
    n_neurons,
    n_latent,
    orthogonal=True,
    add_noise=False,
    noise_std=0.01,
    sparsity=None,
    clusters=None,
    cluster_noise_std=0.1,
    seed=None):
    """
    Generate an emissions matrix with optional orthogonality, noise, sparsity, or clustering with perturbations.

    Parameters:
        n_neurons (int): Number of neurons (rows of emissions matrix).
        n_latent (int): Number of latent dimensions (columns of emissions matrix).
        orthogonal (bool): Whether to make the emissions matrix orthogonal.
        add_noise (bool): Whether to add Gaussian noise to the emissions matrix.
        noise_std (float): Standard deviation of added Gaussian noise.
        sparsity (float, optional): Proportion of elements to zero out (e.g., 0.8 for 80% zeros).
        clusters (int, optional): Number of clusters of neurons projecting similarly.
        cluster_noise_std (float): Standard deviation of noise within clusters.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        emissions (ndarray): Emissions matrix of shape (n_neurons, n_latent).
    """
    if seed is not None:
        np.random.seed(seed)

    if clusters:
        # Generate a clustered emissions matrix with noise
        assert clusters <= n_neurons, "Number of clusters cannot exceed number of neurons."
        neurons_per_cluster = n_neurons // clusters
        C = np.zeros((n_neurons, n_latent))
        for cluster in range(clusters):
            start = cluster * neurons_per_cluster
            end = (cluster + 1) * neurons_per_cluster if cluster < clusters - 1 else n_neurons
            cluster_mean = np.random.randn(1, n_latent)  # Mean weights for cluster
            cluster_noise = np.random.normal(0, cluster_noise_std, (end - start, n_latent))
            C[start:end, :] = cluster_mean + cluster_noise
    else:
        # Generate a random Gaussian emissions matrix
        C = np.random.randn(n_neurons, n_latent)

    if sparsity:
        # Apply sparsity by zeroing out a fraction of the matrix
        mask = np.random.rand(n_neurons, n_latent) > sparsity
        C *= mask

    if orthogonal:
        # Orthogonalize the matrix using SVD
        U, _, Vt = np.linalg.svd(C, full_matrices=False)
        C = U @ Vt

    if add_noise:
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, C.shape)
        C += noise
        if orthogonal:
            # Re-orthogonalize after adding noise
            U, _, Vt = np.linalg.svd(C, full_matrices=False)
            C = U @ Vt
    return C



# Parameters for the bistable system
def simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False):
    """
    Simulate multiple runs of the bistable system:
    dx1/dt = ax1^3 + bx1 + u + eps
    dx2/dt = cx2 + u + eps
    """
    np.random.seed(42)
    a = np.random.uniform(*a_range)
    b = np.random.uniform(*b_range)
    c = np.random.uniform(*c_range)

    x1_runs = np.zeros((num_runs, T))
    x2_runs = np.zeros((num_runs, T))

    for run in range(num_runs):
        for t in range(T - 1):
            eps1 = np.random.normal(0, noise_std)
            eps2 = np.random.normal(0, noise_std)
            if modulate_u:
                u_t=np.sin(u*dt*t*2*np.pi)
            else:
                u_t=u
            x1_runs[run, t + 1] = x1_runs[run, t] + dt * (a * x1_runs[run, t]**3 + b * x1_runs[run, t] + u_t + eps1)
            x2_runs[run, t + 1] = x2_runs[run, t] + dt * (c * x2_runs[run, t] + u_t + eps2)

    # Compute average trajectories
    y_bar = np.array([x1_runs.mean(axis=0), x2_runs.mean(axis=0)])
    return y_bar,x1_runs,x2_runs


def simulate_line_attractor(A, y_bar, dt, T, alpha, u, num_trials=10, noise_std=0.01, fdbk_control=True):
    """
    Simulate a system (line or point attractor) with feedback control to align averages over multiple trials.

    Parameters:
        A: System matrix (n x n).
        y_bar: Target average trajectory (n x T).
        dt: Time step for simulation.
        T: Total time steps.
        alpha: Scaling factor for feedback control.
        u: External input (constant or dynamic, n-dimensional).
        num_trials: Number of trials to simulate.
        noise_std: Standard deviation of noise to add to each trial.
        fdbk_control: Whether to use feedback control to align averages.

    Returns:
        mean_x: Mean trajectory across all trials (n x T).
        all_x: Array of all trial trajectories (num_trials x n x T).
    """
    n = A.shape[0]  # Dimensionality of the system
    all_x = np.zeros((num_trials, n, T))  # To store all trial trajectories

    for trial in range(num_trials):
        x = np.zeros((n, T))  # Initialize state for this trial
        x_bar = np.zeros((n, T))  # Running average state

        for t in range(T - 1):
            # Compute average state up to time t
            x_bar[:, t] = x[:, t]

            if fdbk_control is True:
                # Compute feedback control
                u_tilde = -A @ x_bar[:, t] + (1 / alpha) * (y_bar[:, t + 1] - x_bar[:, t])
            else:
                u_tilde = u

            # Step forward with added noise
            x[:, t + 1] = x[:, t] + dt * (A @ x[:, t] + u_tilde) + np.random.normal(0, noise_std, size=n)

        # Store the trial trajectory
        all_x[trial] = x

    # Compute mean trajectory across trials
    mean_x = np.mean(all_x, axis=0)

    return mean_x, all_x


# Example usage
n_neurons = 100  # Number of neurons
n_latent = 2    # Number of latent dimensions
clusters = None     # Number of clusters
cluster_noise_std = None  # 0.05 Noise within clusters

# Generate emissions matrix with clustering and within-cluster noise
C_clustered_noisy = generate_emissions(
    n_neurons,
    n_latent,
    clusters=clusters,
    add_noise=True,
    noise_std=2.0,
    cluster_noise_std=cluster_noise_std,
    orthogonal=True,
    seed=42
)



T = 1000  # Number of time steps
dt = 0.01  # Time step size
noise_std = 0.05
a_range, b_range, c_range = (-5, -3), (4, 7), (-4, -2)

num_runs = 50  # Number of bistable system runs
alpha = 0.1
u = 0.1
#Line attractor time

#Bistable system
u = 0.3
y_bar, x1, x2 = simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False)
u = -0.3
y_bar_neg, x1_neg, x2_neg = simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False)
pos=np.stack((x1,x2),axis=2)
neg=np.stack((x1_neg,x2_neg),axis=2)

obs_pos= np.einsum('ij,klj->kli', C_clustered_noisy, pos)
obs_neg= np.einsum('ij,klj->kli', C_clustered_noisy, neg)




#% Line attractor system
# Lamb = np.array([[-1, 0], [0, 0]])
# V = np.array([[1, 1], [1, 0]])
# A=V @ Lamb @np.linalg.inv(V)
# u = 0.3
#
# line_x,x_all=simulate_line_attractor(A, y_bar, dt, T, alpha, u, num_trials=50, noise_std=0.0001, fdbk_control=True)
# u = -0.3
#
# line_x_neg,x_all_neg=simulate_line_attractor(A, y_bar_neg, dt, T, alpha, u, num_trials=50, noise_std=0.0001, fdbk_control=True)
# pos=x_all.transpose(0,2,1)
# neg=x_all_neg.transpose(0,2,1)
#
#
# obs_pos= np.einsum('ij,klj->kli', C_clustered_noisy, pos)
# obs_neg= np.einsum('ij,klj->kli', C_clustered_noisy, neg)



# Convert PyTorch tensors to NumPy arrays for TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer
class LeakyRNNCell(tf.keras.layers.Layer):
    """
    Custom RNN cell for simulating leaky integration dynamics with noise.

    Args:
        num_units: Number of hidden units (state dimensionality).
        alpha: Leak rate (0 < alpha <= 1). Larger values make the system less leaky.
        sigma_rec: Standard deviation of recurrent noise (default: 0, no noise).
        activation: Nonlinearity to use (default: 'tanh').
    """
    def __init__(self, num_units, alpha=0.1, sigma_rec=0.0, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.alpha = alpha
        self.sigma_rec = sigma_rec

        # Set activation function
        if activation == "tanh":
            self.activation = tf.tanh
        elif activation == "relu":
            self.activation = tf.nn.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Input weights
        self.kernel_input = self.add_weight(
            shape=(input_dim, self.num_units),
            initializer="glorot_uniform",
            name="kernel_input",
        )

        # Recurrent weights
        self.kernel_recurrent = self.add_weight(
            shape=(self.num_units, self.num_units),
            initializer="orthogonal",
            name="kernel_recurrent",
        )

        # Bias
        self.bias = self.add_weight(
            shape=(self.num_units,),
            initializer="zeros",
            name="bias",
        )

    def call(self, inputs, states):
        prev_state = states[0]

        # Compute input contribution
        input_contribution = tf.matmul(inputs, self.kernel_input)

        # Compute recurrent contribution
        recurrent_contribution = tf.matmul(prev_state, self.kernel_recurrent)

        # Add bias
        total_input = input_contribution + recurrent_contribution + self.bias

        # # Add noise to recurrent dynamics
        # if self.sigma_rec > 0:
        #     noise = tf.random.normal(shape=tf.shape(prev_state), stddev=self.sigma_rec)
        #     total_input += noise

        # Apply nonlinearity
        output = self.activation(total_input)

        # Leaky integration update
        new_state = (1 - self.alpha) * prev_state + self.alpha * output
        # Verify connection to state

        return new_state, [new_state]

x_data_np = np.vstack((0.3 * np.ones((50, 1000, 1)), -0.3 * np.ones((50, 1000, 1))))
y_data_np = np.vstack((obs_pos-obs_pos.mean(axis=0), obs_neg-obs_neg.mean(axis=0)))

# Create TensorFlow Dataset
batch_size = 5
dataset = tf.data.Dataset.from_tensor_slices((x_data_np, y_data_np))
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



sequence_length = x_data_np.shape[1]
input_dim = x_data_np.shape[2]
hidden_units = 3 # Number of RNN hidden units
output_dim = y_data_np.shape[2]  # Dimensionality of the output

# Define model
rnn_layer = tf.keras.layers.RNN(
    LeakyRNNCell(hidden_units, alpha=0.1, sigma_rec=0.0),
    return_sequences=True
)

# Define model
inputs = tf.keras.layers.Input(shape=(sequence_length, input_dim))
hidden_states = rnn_layer(inputs)  # Hidden states as output
outputs = tf.keras.layers.Dense(output_dim)(hidden_states)

model = tf.keras.Model(inputs, outputs)

# Compile and train the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)
history = model.fit(dataset, epochs=50)

batch_size = x_data_np.shape[0]
hidden_state = tf.zeros((batch_size, hidden_units))
trained_rnn_cell = rnn_layer.cell  # This gives the trained LeakyRNNCell
# Simulate hidden state trajectory using trained weights

hidden_states = []
for t in range(sequence_length):
    input_t = x_data_np[:, t, :]  # Input at timestep t
    hidden_state, _ = trained_rnn_cell(input_t, [hidden_state])  # Use the trained cell
    hidden_states.append(hidden_state)

# Stack the hidden states into a tensor
hidden_state_trajectory = tf.stack(hidden_states, axis=1)  # Shape: (batch_size, sequence_length, hidden_units)

# Compute mean of the first 50 timesteps
mean_first_50 = tf.reduce_mean(hidden_state_trajectory[:50, :, :], axis=0)

# Compute mean of the last 50 timesteps
mean_last_50 = tf.reduce_mean(hidden_state_trajectory[-50:,: , :], axis=0)

# Combine these means
combined_means = tf.stack([mean_first_50, mean_last_50], axis=0)


predicted_y = model.predict(x_data_np)



tf.config.run_functions_eagerly(True)  # Ensure eager execution



def rnn_dynamics(inputs, state, leaky_cell):
    """
    Compute the next state of the RNN given inputs and current state.
    """
    next_state, _ = leaky_cell.call(inputs, [state])
    return next_state


def get_zero_state(batch_size, hidden_units):
    """
    Generate a zero initial state.

    Args:
        batch_size: Number of samples in the batch.
        hidden_units: Number of hidden units in the RNN.

    Returns:
        Zero initial state tensor.
    """
    return tf.zeros((batch_size, hidden_units), dtype=tf.float32)


class FixedPointFinderTF:
    def __init__(self, leaky_cell, input_dim, hidden_units):
        self.leaky_cell = leaky_cell
        self.input_dim = input_dim
        self.hidden_units = hidden_units

    def find_fixed_points(self, initial_states, inputs, max_iter=100, tol=1e-6):
        """
        Find fixed points of the RNN dynamics.

        Args:
            initial_states: Initial states to start the optimization, shape (batch_size, hidden_units).
            inputs: Input tensor, shape (batch_size, input_dim). Can be None.
            max_iter: Maximum number of optimization iterations.
            tol: Tolerance for convergence.

        Returns:
            fixed_points: Fixed points found, shape (batch_size, hidden_units).
            converged: Boolean indicating if the fixed points converged.
        """
        converged = False
        state = tf.Variable(initial_states, trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        for i in range(max_iter):
            with tf.GradientTape() as tape:
                # Handle the case where inputs is None
                next_state = rnn_dynamics(inputs, state, self.leaky_cell)
                loss = tf.reduce_mean(tf.square(next_state - state))  # Loss: ||next_state - state||

            gradients = tape.gradient(loss, [state])
            optimizer.apply_gradients(zip(gradients, [state]))

            if loss.numpy() < tol:  # Eager execution allows .numpy()
                print(f"Converged after {i + 1} iterations with loss: {loss.numpy()}")
                converged = True
                break

        return state, converged

    def compute_jacobian(self, inputs, fixed_points):
        """
        Compute the Jacobian matrix of the RNN dynamics at the fixed points.

        Args:
            inputs: Input tensor, shape (batch_size, input_dim). Can be None.
            fixed_points: Tensor of fixed points, shape (batch_size, hidden_units).

        Returns:
            jacobians: List of Jacobian matrices, one for each fixed point.
        """
        batch_size = fixed_points.shape[0]
        jacobians = []

        for i in range(batch_size):
            single_fixed_point = tf.Variable(fixed_points[i:i + 1], trainable=True)

            with tf.GradientTape() as tape:
                tape.watch(single_fixed_point)
                # Handle the case where inputs is None
                next_state = rnn_dynamics(inputs[i:i + 1], single_fixed_point, self.leaky_cell)

            jacobian = tape.jacobian(next_state, single_fixed_point)
            if jacobian is not None:
                jacobians.append(jacobian.numpy())  # Convert to NumPy for further processing
            else:
                raise ValueError(f"Jacobian calculation failed for fixed point {i}. Verify the computational graph.")

        return jacobians

    def analyze_stability(self, jacobians):
        """
        Analyze stability of the fixed points using Jacobian eigenvalues.

        Args:
            jacobians: List of Jacobian matrices.

        Returns:
            stability: List of stability results. Each entry contains:
                       - Eigenvalues of the Jacobian
                       - Boolean indicating stability (True if all eigenvalues have magnitude < 1)
        """
        stability_results = []
        for jacobian in jacobians:
            eigenvalues = np.linalg.eigvals(jacobian)
            is_stable = np.all(np.abs(eigenvalues) < 1)  # Stability condition for discrete systems
            stability_results.append((eigenvalues, is_stable))
        return stability_results

# Define LeakyRNNCell and FixedPointFinder
leaky_cell=trained_rnn_cell
# Manually build the LeakyRNNCell to initialize weights
input_dim = 1  # Dimensionality of the inputs

# Create FixedPointFinder
fpf = FixedPointFinderTF(leaky_cell, input_dim=input_dim, hidden_units=hidden_units)

# Generate random inputs and initial states
# Number of points to sample from each row
samples_per_row = 250

# For each row, generate random indices
batch_size = combined_means.shape[1]  # 1000
random_indices_0 = tf.random.uniform(shape=(samples_per_row,), minval=0, maxval=batch_size, dtype=tf.int32)
random_indices_1 = tf.random.uniform(shape=(samples_per_row,), minval=0, maxval=batch_size, dtype=tf.int32)

# Sample 250 points from the first row
sampled_from_first_row = tf.gather(combined_means[0], random_indices_0, axis=0)  # Shape: (250, 10)

# Sample 250 points from the second row
sampled_from_second_row = tf.gather(combined_means[1], random_indices_1, axis=0)  # Shape: (250, 10)

# Combine the samples into a single tensor
initial_states = tf.concat([sampled_from_first_row, sampled_from_second_row], axis=0)  # Shape: (500, 10)

# Create a tensor of size (250, 1) with all values as 0.3
first_250 = tf.fill((samples_per_row, 1), 0.3)

# Create a tensor of size (250, 1) with all values as -0.3
last_250 = tf.fill((samples_per_row, 1), -0.3)

# Concatenate the two tensors along the first dimension
inputs = tf.concat([first_250, last_250], axis=0)  # Shape: (500, 1)

# Find fixed points
from sklearn.decomposition import PCA
good=[]
fps=[]
for i in range(len(initial_states)):

    single_initial_state = initial_states[i:(i+1)]  # First element in batch
    single_input = inputs[i:(i+1)]
    single_fixed_point,conver = fpf.find_fixed_points(single_initial_state, single_input,tol=10e-5)
    good.append(conver)
    fps.append(single_fixed_point.numpy()[0])










pca=PCA(n_components=2)
pca.fit(combined_means[0].numpy()-combined_means[1].numpy())


isgood1=np.where(np.array(good)[0:250]==True)[0]
isgood2=np.where(np.array(good)[250:]==True)[0]

plt.plot(pca.components_[0,:]@combined_means[0].numpy().transpose())
plt.plot(random_indices_0.numpy()[isgood1],pca.components_[0,:]@np.array(fps)[0:250][isgood1,:].transpose(),'go')
plt.plot(pca.components_[0,:]@combined_means[1].numpy().transpose())
plt.plot(random_indices_1.numpy()[[isgood2],pca.components_[0,:]@np.array(fps)[250:].transpose()[isgood2],'bo')


plt.show()

traj_x=np.arange(0,1000,1)

plt.plot(traj_x,combined_means[0].numpy())
plt.plot(random_indices_0.numpy(),np.array(fps)[0:250],'o')
plt.show()
plt.plot(traj_x,combined_means[1].numpy())
plt.plot(random_indices_1.numpy(),np.array(fps)[250:],'o')

plt.show()


# For stability
# jacobians=fpf.compute_jacobian(inputs,fixed_points)
# eigy=[]
# for i in range((fixed_points.shape[0])):
#     simplified_jacobian = tf.squeeze(jacobians[i])  # Removes dimensions of size 1
#     eigy.append(np.linalg.eig(simplified_jacobian.numpy())[0])

















##############

#
#
#
#
#
# T = 1000  # Number of time steps
# dt = 0.01  # Time step size
# noise_std = 0.05
# a_range, b_range, c_range = (-5, -3), (4, 7), (-4, -2)
#
# num_runs = 50  # Number of bistable system runs
# alpha = 0.1
# u = 0.3
#
# y_bar,x1,x2 = simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False)
# u = -0.3
# # Simulate bistable system
# y_bar_neg,x1neg,x2neg = simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False)
#
#
# #% Line attractor system
# Lamb = np.array([[-1, 0], [0, 0]])
# V = np.array([[1, 1], [1, 0]])
# A=V @ Lamb @np.linalg.inv(V)
#
# u = 0.3
#
# line_x,x_all=simulate_line_attractor(A, y_bar, dt, T, alpha, u, num_trials=50, noise_std=0.0001, fdbk_control=True)
# u = -0.3
#
# line_x_neg,x_all_neg=simulate_line_attractor(A, y_bar_neg, dt, T, alpha, u, num_trials=50, noise_std=0.0001, fdbk_control=True)
#
#
# pos=x_all.transpose(0,2,1)
# neg=x_all_neg.transpose(0,2,1)
#
#
# obs_pos= np.einsum('ij,klj->kli', C_clustered_noisy, pos)
# obs_neg= np.einsum('ij,klj->kli', C_clustered_noisy, neg)
#
#
# dat=[]
# inputs=[]
# data = obs_pos.mean(axis=0)+np.random.random((1000,100))*0.001
# # data=(C_clustered_noisy @ y_bar).transpose()+np.random.random((1000,100))*0.001
# for i in range(50):
#     inputs.append(0.3*np.ones((1000,1)))
#     dat.append(np.random.random((1000,100))*0.001+obs_pos[i,:,:]-obs_pos.mean(axis=0))
#
# for i in range(50):
#     inputs.append(-0.3 * np.ones((1000, 1)))
#     dat.append(np.random.random((1000, 100)) * 0.001 + obs_neg[i, :, :] - obs_neg.mean(axis=0))
#
#
#
# #Shitty attempt at residual hankel
#
# import numpy as np
# from scipy.linalg import hankel, svd
#
# def compute_aligned_residuals(single_trials, condition_averages):
#     """ Subtract condition averages from single-trial data to compute residuals. """
#     return single_trials - condition_averages
#
# def construct_hankel_matrix(residuals, lags=5):
#     """ Construct a reduced-size Hankel matrix for given residuals using specified lags. """
#     # Reduce the size by only considering a window of residuals
#     window_size = 25  # adjust based on your dataset size
#     return hankel(residuals[:window_size, :lags], residuals[:window_size, lags-1:])
#
# def dynamics_subspace_estimation(residuals, n_components=5):
#     """ Estimate the dynamics subspace using randomized SVD for efficiency. """
#     H = construct_hankel_matrix(residuals)
#     U, s, Vh = randomized_svd(H, n_components=n_components)
#     return U
#
# def estimate_residual_latent_state(residuals, dynamics_subspace):
#     """ Project residuals onto the dynamics subspace to estimate latent states. """
#     return  dynamics_subspace @ residuals
#
# import statsmodels.api as sm
#
# def estimate_dynamics_matrices(latent_states, next_latent_states, regularization_alpha):
#     """ Estimate dynamics matrices using a penalized least squares with regularization. """
#     # Here we use OLS for simplicity, could be extended to include regularization.
#     model = sm.OLS(next_latent_states, sm.add_constant(latent_states))
#     results = model.fit_regularized(alpha=regularization_alpha)
#     return results.params
#
#
# def reduce_dimensionality(residuals, n_components=10):
#     """ Reduce the dimensionality of residuals using SVD. """
#     U, s, Vt = svd(residuals, full_matrices=False)
#     return U[:, :n_components] * s[:n_components]  # Return projected data
#
# def reshape_data(residuals):
#     """ Reshape the (trials, time, features) array to (trials, time * features) for SVD. """
#     n_trials, n_time, n_features = residuals.shape
#     return residuals.reshape(n_trials, n_time * n_features)
#
#
# obs_pos=obs_pos+np.random.random((50,1000,100))*0.001
#
#
# #step (i)
# U, s, Vt = svd(np.vstack((obs_pos.mean(axis=0),obs_neg.mean(axis=0))), full_matrices=False)
# # Step (ii)
# residuals = compute_aligned_residuals(obs_pos, obs_pos.mean(axis=0))
#
# resres=np.zeros((50,1000,5))
# for i in range(residuals.shape[0]):
#     resres[i,:,:]=np.dot(residuals[0,:,:],Vt[:,:5])
#
#
# dynamics_subspace = dynamics_subspace_estimation(resres,5)
# # Step (iii)
# #TODO: broken
# latent_states = estimate_residual_latent_state(resres, dynamics_subspace[:, :5])  # using first 5 singular vectors
#
#
# #############3
#
# T = 1000  # Number of time steps
# dt = 0.01  # Time step size
# noise_std = 0.05
# a_range, b_range, c_range = (-5, -3), (4, 7), (-4, -2)
#
# num_runs = 50  # Number of bistable system runs
# alpha = 0.1
# u = 0.1
#
# y_bar,x1,x2 = simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False)
# u = -0.1
# # Simulate bistable system
# y_bar_neg,x1neg,x2neg = simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False)
#
#
#
# #% Line attractor system
# Lamb = np.array([[-1, 0], [0, 0]])
# V = np.array([[1, 1], [1, 0]])
# A=V @ Lamb @np.linalg.inv(V)
#
#
# line_x,x_all=simulate_line_attractor(A, y_bar, dt, T, alpha, u, num_trials=50, noise_std=0.05, fdbk_control=True)
# line_x_neg,x_all_neg=simulate_line_attractor(A, y_bar_neg, dt, T, alpha, u, num_trials=50, noise_std=0.05, fdbk_control=True)
#
# # Example usage
# n_neurons = 100  # Number of neurons
# n_latent = 2    # Number of latent dimensions
# clusters = None     # Number of clusters
# cluster_noise_std = None  # 0.05 Noise within clusters
#
# # Generate emissions matrix with clustering and within-cluster noise
# C_clustered_noisy = generate_emissions(
#     n_neurons,
#     n_latent,
#     clusters=clusters,
#     add_noise=True,
#     noise_std=2.0,
#     cluster_noise_std=cluster_noise_std,
#     orthogonal=True,
#     seed=42
# )
#
#
# plt.plot((C_clustered_noisy @ y_bar).transpose())
# plt.plot((C_clustered_noisy @ y_bar_neg).transpose())
#
#
# #Do a PCA and transform to 1D space
# from sklearn.decomposition import PCA
# pca=PCA(n_components=2)
# pca.fit(((C_clustered_noisy @ line_x)-(C_clustered_noisy @ line_x_neg)).transpose())
# W=pca.components_
# y1=(C_clustered_noisy @ line_x)
# y2=(C_clustered_noisy @ line_x_neg)
# ylow_1=W[0,:] @y1
# ylow_2=W[0,:] @y2
#
#
# #Bistable system
# u = 0.3
# y_bar, x1, x2 = simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False)
# u = -0.3
# y_bar_neg, x1_neg, x2_neg = simulate_bistable_system(a_range, b_range, c_range, noise_std, dt, T, u, num_runs,modulate_u=False)
# pos=np.stack((x1,x2),axis=2)
# neg=np.stack((x1_neg,x2_neg),axis=2)
#
# obs_pos= np.einsum('ij,klj->kli', C_clustered_noisy, pos)
# obs_neg= np.einsum('ij,klj->kli', C_clustered_noisy, neg)
#
#
#
# #
# # Create DataLoader
# y_data=torch.tensor(np.vstack((obs_pos,obs_neg))).float()
# x_data=torch.tensor(np.vstack((0.2*np.ones((50,1000,1)),-0.2*np.ones((50,1000,1))))).float()
#
#
# dataset = TensorDataset(x_data, y_data)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
#
#
#
# # Initialize RNN, optimizer, and loss function
# hidden_size=8
# rnn = NNtorch.ContinuousRNN_basic(1, hidden_size, 100,dt=dt,nonlinearity=torch.tanh)
# optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()
#
# # Training loop
# epochs = 150
# epoch_losses = []
#
# for epoch in range(epochs):
#     epoch_loss = 0
#     for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
#         batch_size, T, _ = x_batch.size()
#
#         # Initialize hidden states for the batch
#         h = torch.zeros(batch_size, hidden_size)
#
#         optimizer.zero_grad()
#         batch_loss = 0
#
#         # Process each time step in the batch
#         for t in range(T):
#             x_t = x_batch[:, t, :]  # Input at time t: [batch_size, input_size]
#             y_t = y_batch[:, t, :]  # Target at time t: [batch_size, output_size]
#             y_pred, h = rnn(x_t, h)  # Forward pass
#             batch_loss += loss_fn(y_pred, y_t)
#
#         batch_loss.backward()  # Backpropagation
#         optimizer.step()  # Parameter update
#
#         epoch_loss += batch_loss.item()
#  # Print running loss for the current batch
#
#     # Print average loss for the epoch
#     print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(dataloader):.4f}")
#
#
# #Simulate
# h = torch.zeros(hidden_size)  # Initial hidden state
# trajectory = []
#
# x=x_data[0]
# for t in range(T):
#     x_t = x[t]  # Input at time t
#     h = h + dt * (-h + F.tanh(torch.matmul(h, rnn.W_h.T) + torch.matmul(x_t, rnn.W_x.T) + rnn.b))  # Dynamics
#     trajectory.append(h.detach().numpy())
#
# trajectory = np.array(trajectory)
#
#
# h = torch.zeros(hidden_size)  # Initial hidden state
# trajectoryb = []
# x=x_data[99]
# for t in range(T):
#     x_t = x[t]  # Input at time t
#     h = h + dt * (-h + F.tanh(torch.matmul(h, rnn.W_h.T) + torch.matmul(x_t, rnn.W_x.T) + rnn.b))  # Dynamics
#     trajectoryb.append(h.detach().numpy())
# trajectoryb = np.array(trajectoryb)
#
# traj=np.vstack((trajectory,trajectoryb))
#
# pca.fit((trajectory-trajectoryb))
#
# plt.plot(pca.components_[0,:]@trajectory.transpose())
# plt.plot(pca.components_[0,:]@trajectoryb.transpose())
#
# plt.show()
#
#
# #Find fixed points:
#
# # initial_guesses = torch.tensor(np.vstack((np.vstack((trajectory[0:10,:],trajectory[500:510,:])),np.vstack((trajectoryb[0:10,:],trajectoryb[500:510,:]))))).float()
# initial_guesses = torch.tensor(np.vstack((trajectory[np.arange(0,500,5),:],trajectoryb[np.arange(0,500,5),:]))).float()
# inputs=torch.zeros_like(initial_guesses)
# inputs[0:int(initial_guesses.shape[0]/2),:]=0.1
# inputs[int(initial_guesses.shape[0]/2):,:]=-0.1
# inputs=inputs[:,0]
#
# fp,stability=find_fixed_points_with_nullspace(rnn, initial_guesses, inputs=inputs, tolerance=10e-5, max_iter=1000, verbose=True)
#
# index_values = [d['index'] for d in stability]
# fps = [d['Fixed Point'] for d in stability]
# eig = [d['Eigenvalues'] for d in stability]
#
# x_index=np.hstack((np.arange(0,500,5),np.arange(0,500,5)))[index_values]
#
# plt.plot(np.arange(0,1000,1),pca.components_[0,:]@trajectory.transpose())
# plt.plot(np.arange(0,1000,1),pca.components_[0,:]@trajectoryb.transpose())
#
# plt.plot(x_index,pca.components_[0,:] @ np.array(fps).transpose(),'o')
#
# tp=pca.components_[0,:] @ np.array(fps).transpose()
#
#
#
#
# #Line attractor time
# u = 0.3
#
# line_x,x_all=simulate_line_attractor(A, y_bar, dt, T, alpha, u, num_trials=50, noise_std=0.001, fdbk_control=True)
# u = -0.3
#
# line_x_neg,x_all_neg=simulate_line_attractor(A, y_bar_neg, dt, T, alpha, u, num_trials=50, noise_std=0.001, fdbk_control=True)
# pos=x_all.transpose(0,2,1)
# neg=x_all_neg.transpose(0,2,1)
#
#
# obs_pos= np.einsum('ij,klj->kli', C_clustered_noisy, pos)
# obs_neg= np.einsum('ij,klj->kli', C_clustered_noisy, neg)
#
#
#
# #
# # Create DataLoader
# y_data=torch.tensor(np.vstack((obs_pos-np.mean(obs_pos,axis=0),obs_neg-np.mean(obs_neg,axis=0)))).float()
# x_data=torch.tensor(np.vstack((0.3*np.ones((50,1000,1)),-0.3*np.ones((50,1000,1))))).float()
#
#
# dataset = TensorDataset(x_data, y_data)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
#
#
#
# # Initialize RNN, optimizer, and loss function
# hidden_size=3
# rnn = NNtorch.ContinuousRNN_basic(1, hidden_size, 100,dt=dt,nonlinearity=torch.tanh)
# optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()
#
# # Training loop
# epochs = 50
# epoch_losses = []
#
# for epoch in range(epochs):
#     epoch_loss = 0
#     for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
#         batch_size, T, _ = x_batch.size()
#
#         # Initialize hidden states for the batch
#         h = torch.zeros(batch_size, hidden_size)
#
#         optimizer.zero_grad()
#         batch_loss = 0
#
#         # Process each time step in the batch
#         for t in range(T):
#             x_t = x_batch[:, t, :]  # Input at time t: [batch_size, input_size]
#             y_t = y_batch[:, t, :]  # Target at time t: [batch_size, output_size]
#             y_pred, h = rnn(x_t, h)  # Forward pass
#             batch_loss += loss_fn(y_pred, y_t)
#
#         batch_loss.backward()  # Backpropagation
#         optimizer.step()  # Parameter update
#
#         epoch_loss += batch_loss.item()
#  # Print running loss for the current batch
#
#     # Print average loss for the epoch
#     print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(dataloader):.4f}")
#
#
# #Simulate
# h = torch.zeros(hidden_size)  # Initial hidden state
# trajectory = []
#
# x=x_data[0]
# for t in range(T):
#     x_t = x[t]  # Input at time t
#     h = h + dt * (-h + F.tanh(torch.matmul(h, rnn.W_h.T) + torch.matmul(x_t, rnn.W_x.T) + rnn.b))  # Dynamics
#     trajectory.append(h.detach().numpy())
#
# trajectory = np.array(trajectory)
#
#
# h = torch.zeros(hidden_size)  # Initial hidden state
# trajectoryb = []
# x=x_data[99]
# for t in range(T):
#     x_t = x[t]  # Input at time t
#     h = h + dt * (-h + F.tanh(torch.matmul(h, rnn.W_h.T) + torch.matmul(x_t, rnn.W_x.T) + rnn.b))  # Dynamics
#     trajectoryb.append(h.detach().numpy())
# trajectoryb = np.array(trajectoryb)
#
# traj=np.vstack((trajectory,trajectoryb))
#
# pca.fit((trajectory-trajectoryb))
#
# plt.plot(pca.components_[0,:]@trajectory.transpose())
# plt.plot(pca.components_[0,:]@trajectoryb.transpose())
#
# plt.show()
#
#
#
# initial_guesses = torch.tensor(np.vstack((trajectory[np.arange(0,500,5),:],trajectoryb[np.arange(0,500,5),:]))).float()
#
#
# inputs=torch.zeros_like(initial_guesses)
# inputs[0:100,:]=0.3
# inputs[100:,:]=-0.3
# inputs=inputs[:,0]
#
# fp,stability=find_fixed_points_with_nullspace(rnn, initial_guesses, inputs=None, tolerance=10e-6, max_iter=1000, verbose=True)
# index_values = [d['index'] for d in stability]
# fps = [d['Fixed Point'] for d in stability]
#
# x_index=np.hstack((np.arange(0,500,5),np.arange(0,500,5)))[index_values]
#
# plt.plot(np.arange(0,1000,1),pca.components_[0,:]@trajectory.transpose())
# plt.plot(np.arange(0,1000,1),pca.components_[0,:]@trajectoryb.transpose())
#
# plt.plot(x_index,pca.components_[0,:] @ np.array(fps).transpose(),'o')
#
# tp=pca.components_[0,:] @ np.array(fps).transpose()
#



#### TENSORFLOW
