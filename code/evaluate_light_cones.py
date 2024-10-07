from collections.abc import Iterable
import os

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info import StabilizerState, Pauli


def estimate_light_cone(qc: QuantumCircuit,
                        O: Iterable,
                        K: int,
                        dir: str,
                        get_loss:  bool = False):

    """ Uses Monte Carlo sampling and the Clifford tableau to estimate the min/mean/max lightcone
        for the given ansatz.

        Args:
            qc:         parameterized ansatz given as quantum circuit
            O:          observable as sparse list of form: [(index, label), ...)],
                        where index determines the qubit and label should be 'X', 'Y', or 'Z'
            K:          number of Monte Carlo samples
            dir:        directory to store the results in
            get_loss:   compute loss values and variance

    """
    # define circuit corresponding to observable
    qc_h = QuantumCircuit(qc.num_qubits)
    for (i, h) in O:
        if h == 'X':
            qc_h.x(i)
        if h == 'Y':
            qc_h.y(i)
        if h == 'Z':
            qc_h.z(i)
    # Pauli representation of the Hamiltonian
    H_pauli = Pauli(qc_h)

    # number of parameters and set of possible parameter values
    m = qc.num_parameters
    param_set = [0, np.pi / 2]

    # sample parameter values K-times and determine corresponding lightcone
    lightcone_pauli = []

    for k in range(K):
        # sample parameter values
        params = np.random.choice(param_set, m)
        qc_assigned = qc.assign_parameters(params)
        if isinstance(H_pauli, Iterable):
            H_evolved = []
            for h in H_pauli:
                H_evolved.append(h.evolve(Clifford(qc_assigned), frame='h'))
        else:
            H_evolved = H_pauli.evolve(Clifford(qc_assigned), frame='h')

        lightcone_pauli.append(sum(H_evolved.x | H_evolved.z))
    if not os.path.exists(dir + '/mean.npy'):
        np.save(dir + '/mean.npy', [np.mean(lightcone_pauli)])
        np.save(dir + '/lightcones.npy', [lightcone_pauli])
    else:
        mean_temp = np.load(dir + '/mean.npy')
        lcs_temp = np.load(dir + '/lightcones.npy')
        np.save(dir + '/mean.npy', np.append(mean_temp, np.mean(lightcone_pauli)))
        np.save(dir + '/lightcones.npy', np.append(lcs_temp, lightcone_pauli))

    if get_loss:
        variance_loss = 0
        for k in range(K * 10): # Use 10 times more samples
            # sample parameter values
            params = np.random.choice(param_set, m)
            qc_assigned = qc.assign_parameters(params)
            state = StabilizerState(qc_assigned)
            exp_val = state.expectation_value(H_pauli)
            variance_loss += exp_val**2 / (K * 10)
        if not os.path.exists(dir + '/var_loss.npy'):
            np.save(dir + '/var_loss.npy', [variance_loss])
        else:
            var_temp = np.load(dir + '/var_loss.npy')
            np.save(dir + '/var_loss.npy', np.append(var_temp, [variance_loss]))

    return
