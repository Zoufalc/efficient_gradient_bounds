# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools
import numpy as np
from scipy.stats import multivariate_normal

import torch
from torch import nn

from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_algorithms.gradients.spsa.spsa_sampler_gradient import SPSASamplerGradient
from qiskit_algorithms.gradients.reverse.reverse_gradient import ReverseEstimatorGradient
from qiskit_algorithms.gradients.param_shift.param_shift_sampler_gradient import ParamShiftSamplerGradient

from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Sampler as AerSampler


# from qiskit.utils import algorithm_globals
seed = 42
# algorithm_globals.random_seed = seed
np.random.seed(seed)
torch.manual_seed(seed)


""" Initial Distribution """
def generate_gaussian_mixture(mus, sigs, num_qubits, box=2):
    grain = 2**(num_qubits//2)
    num_gaussians = len(mus)
    x = np.linspace(-box, box, grain)
    y = np.linspace(-box, box, grain)
    X, Y = np.meshgrid(x,y)
    grid = np.empty(X.shape + (2,))
    grid[:, :, 0] = X; grid[:, :, 1] = Y
    Z = 0
    for i in range(num_gaussians):
        rv = multivariate_normal(mus[i], sigs[i])
        Z += rv.pdf(grid)
    return X, Y, Z/np.sum(Z), grid.reshape(-1, 2)


""" Classical Discriminator """
# Discriminator input (efficient enumeration of {0, 1}^n)
def unpack(n):
    x = np.arange(2**n).reshape([-1, 1])
    mask = 2**np.arange(n, dtype=x.dtype).reshape([1, n])
    return (x & mask).astype(bool).astype(int).reshape([2**n, n]).T

# Custom discriminator initialisation (not currently used)
def init_weights(m):
    if isinstance(m, nn.Linear):
        std = np.sqrt(4 / m.weight.size(1))
        torch.nn.init.normal_(m.weight, std=std)
        torch.nn.init.normal_(m.bias, std=0.1)

# Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=15, leaky_gamma=0.05): #hidden_size=64
        super().__init__()
        
        self.sequence = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(leaky_gamma),
            nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(leaky_gamma),
            # nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(leaky_gamma),
            nn.Linear(hidden_size, output_size)
        )

        # self.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.sequence(input)
        return x

# Build (sparse) Z-observables from list of alphas
def build_obs(alphas, sparse=False):
    if len(alphas) == 0:
        return []
    num_qubits = len(alphas[0])
    Z_alphas = [("Z"*int(sum(alphas[j])), [i for i in range(num_qubits) if alphas[j][i] == 1], 1) for j in range(len(alphas))]
    obs = [SparsePauliOp.from_sparse_list([Z_alpha], num_qubits=num_qubits) for Z_alpha in Z_alphas]
    if sparse:
        obs = [ob.to_matrix(sparse=True) for ob in obs]
    return obs

# Build Z-observables from list of alphas (only paulis to avoid memory overload)
def build_obs_paulis(alphas):
    if len(alphas) == 0:
        return []
    num_qubits = len(alphas[0])
    Z_alphas = [("Z" * int(sum(alphas[j])), [i for i in range(num_qubits) if alphas[j][i] == 1], 1) for j in
                range(len(alphas))]
    obs = [SparsePauliOp.from_sparse_list([Z_alpha], num_qubits=num_qubits) for Z_alpha in Z_alphas]
    return obs

""" Quantum Generator """
# Generator with TorchConnector (forward sampling + backward gradient computed with SPSA)
def Generator(qc, input_weights, input_params, shots=10000, seed=42, grad='spsa') -> TorchConnector:
    sampler = Sampler(options={"shots": shots}) if shots > 0 else Sampler()
    epsilon = 0.000001 if shots == 0 else 0.1 # 1/np.sqrt(shots)
    if grad=='spsa':
        sampler_gradient = SPSASamplerGradient(sampler, epsilon=epsilon, batch_size=10, seed=seed)
    elif grad=='paramshift':
        sampler_gradient = ParamShiftSamplerGradient(sampler, seed=seed)
    else:
        raise ValueError('Gradient Type not supported')
    if shots <= 100000: # More efficient forward pass using Aer
        sampler = AerSampler(run_options={"shots": shots})

    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        gradient=sampler_gradient,
        input_params=input_params,
        weight_params=input_weights
    )



    initial_weights = np.random.uniform(-np.pi, np.pi, len(input_weights))

    return TorchConnector(qnn, initial_weights)

# Estimator Generator (efficient & exact way to compute gradients)
# def Generator_Estimator(qc, obs):
#     estimator = Estimator()
#     estimator_gradient = ReverseEstimatorGradient()
#
#     qnn = EstimatorQNN(
#         circuit=qc,
#         observables=obs,
#         estimator=estimator,
#         gradient=estimator_gradient,
#         input_params=[],
#         weight_params=qc.parameters
#     )
#
#     return qnn


""" Cross-entropy loss """
def f_loss(probs, D_x, wasserstein=False):
    ep = 1e-6 # To avoid torch.log(very small number)) = -infinity
    D_x = (-1)*D_x if wasserstein else -torch.log(torch.sigmoid(D_x)+ep)
    return torch.sum(probs*D_x)

def f_tilde_loss(probs, D_x, wasserstein=False):
    ep = 1e-6 # To avoid torch.log(very small number)) = -infinity
    D_x = D_x if wasserstein else -torch.log(1-torch.sigmoid(D_x)+ep)
    return torch.sum(probs*D_x)


""" Alpha Gradients """
# Build alphas of order k (n qubits)
def get_alphas(n, orders=None):
    bit_strings = [list(seq) for seq in itertools.product([0, 1], repeat=n)]
    bit_strings.sort(key=lambda x: x.count(1))
    bit_strings = np.array(bit_strings)
    if orders is not None:
        return bit_strings[np.isin(np.sum(bit_strings, axis=1), orders)]
    else:
        return bit_strings

# Build (sparse) Z-observables from list of alphas
def build_obs(alphas, sparse=False):
    num_qubits = len(alphas[0])
    Z_alphas = [("Z"*int(sum(alphas[j])), [i for i in range(num_qubits) if alphas[j][i] == 1], 1) for j in range(len(alphas))]
    obs = [SparsePauliOp.from_sparse_list([Z_alpha], num_qubits=num_qubits) for Z_alpha in Z_alphas]
    if sparse:
        obs = [ob.to_matrix(sparse=True) for ob in obs]
    return obs

# Evaluate the probability distribution closest to a given quasi-probability distribution
def nearest_probability_distribution(quasi_dist):
    """Takes a quasiprobability distribution and maps
    it to the closest probability distribution as defined by
    the L2-norm.

    Parameters:
        quasi_dist (array): Quasi probability distribution

    Returns:
        new_probs: Nearest probability distribution.

    Notes:
        Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
    """
    sorted_probs = quasi_dist
    num_elems = len(sorted_probs)
    new_probs = []
    beta = 0
    diff = 0
    for val in sorted_probs:
        temp = val + beta / num_elems
        if temp < 0:
            beta += val
            num_elems -= 1
            diff += val * val
            new_probs.append(0)
        else:
            diff += (beta / num_elems) * (beta / num_elems)
            new_probs.append(val + beta / num_elems)
    return new_probs


# Estimator Generator (efficient & exact way to compute gradients)
def Generator_Estimator(qc, obs):
    estimator = Estimator()
    estimator_gradient = ReverseEstimatorGradient()

    qnn = EstimatorQNN(
        circuit=qc,
        observables=obs,
        estimator=estimator,
        gradient=estimator_gradient,
        input_params=[],
        weight_params=qc.parameters
    )

    return qnn