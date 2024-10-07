"""
This file demonstrates how to evaluate the light cones for a given observable with respect to a particular ansatz
and how to plot the resulting gradient bounds.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import NLocal

from code.evaluate_light_cones import *
from code.utilities.plot_light_cones import plot_gradient_bounds

# set number of qubits
num_qubits = list(range(5,8))

# set ansatz repetitions to be evaluated
reps = range(1, 4)

# number of random samples
K = 1000
dir = 'test_dir/'


# run estimation
mean_lcs = []
variance_loss = []
k_list =[]
for i, n in enumerate(num_qubits):
    k_list.append(int(np.log(n)))
    r = reps[i]
    H = []
    # define observable locality
    k = range(int(np.log(n)))
    for locality in k:
        H.append((locality, 'Z'))

    # ------Define ansatz------
    # EfficientSU2
    # qc_effsu2 = EfficientSU2(n, reps=reps, entanglement='pairwise')

    # Cartan building blocks
    cartan_qc = QuantumCircuit(2)
    param0 = Parameter('phi' + str(0))
    param1 = Parameter('phi' + str(1))
    param2 = Parameter('phi' + str(2))
    param3 = Parameter('phi' + str(3))
    param4 = Parameter('phi' + str(4))
    param5 = Parameter('phi' + str(5))
    param6 = Parameter('phi' + str(6))
    param7 = Parameter('phi' + str(7))
    param8 = Parameter('phi' + str(8))
    param9 = Parameter('phi' + str(9))
    param10 = Parameter('phi' + str(10))
    param11 = Parameter('phi' + str(11))
    param12 = Parameter('phi' + str(12))
    param13 = Parameter('phi' + str(13))
    param14 = Parameter('phi' + str(14))
    cartan_qc.u(param0, param1, param2, 0)
    cartan_qc.u(param3, param4, param5, 1)
    cartan_qc.rxx(param6, 0, 1)
    cartan_qc.ryy(param7, 0, 1)
    cartan_qc.rzz(param8, 0, 1)
    cartan_qc.u(param9, param10, param11, 0)
    cartan_qc.u(param12, param13, param14, 1)

    entanglement = [(i, (i + 1)) for i in range(0, int(n)-1, 2)] + [(i, (i + 1)) for i in range(1, int(n)-1, 2)]
    qc_cartan = NLocal(n, None, cartan_qc, entanglement=entanglement, reps=r)

    lcs = estimate_light_cone(qc_cartan, H, K, dir=dir, get_loss=True)


# plot results
plot_gradient_bounds(dir, k_list, num_qubits, reps, K, plot_variance=True)
