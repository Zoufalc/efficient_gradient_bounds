import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

import seaborn as sns
sns.set_theme(style="white", palette="colorblind")
plt.rc('font', size=30)
plt.rc('axes', titlesize=40, titlepad=32, labelsize=46, labelpad=30)
plt.rc('xtick', labelsize=40)
plt.rc('ytick', labelsize=40)
plt.rc('legend', fontsize=40)
plt.rc('lines', linewidth=6)


def plot_gradient_bounds(dir: str,
                         k_list: int,  #locality
                         num_qubits: Iterable,
                         reps: Iterable,
                         K: int,
                         plot_variance = False
                         ):
    """
    Plot gradient bounds based on light cones
    Args:
        dir: Directory to get the data from and store the plots to
        k_list: Observable locality
        num_qubits: Number of system qubits
        reps: Number of repetitions in the ansatz
        K: Number of samples
    """

    #Load arrays
    lightcones = np.load(dir + 'lightcones.npy')
    if plot_variance:
        variance_loss = np.load(dir + '/var_loss.npy')
    upper_bound = []
    lower_bound = []
    est_error = []
    est_error_upper_bound = []
    est_error_lower_bound = []

    plt.figure(figsize=(30, 20))
    for i in range(int(len(lightcones)/K)):
        try:
            est_error_upper_bound += [1.96 * np.std(np.power(1 / 2, lightcones[i * K: (i + 1) * K])) / np.sqrt(K)]
            est_error_lower_bound += [1.96 * np.std(np.power(1 / 4, lightcones[i * K: (i + 1) * K])) / np.sqrt(K)]
            lower_bound += [np.mean(np.power(1 / 4, lightcones[i * K: (i + 1) * K]))]
            upper_bound += [np.mean(np.power(1 / 2, lightcones[i * K: (i + 1) * K]))]
            est_error += [1.96 * np.std(lightcones[i*K: (i+1)*K]) / np.sqrt(K)]  # 0.95
            counts, bins = np.histogram(lightcones[i*K: (i+1)*K])
            plt.stairs(counts, bins, label='num qubits '+str(num_qubits[i]))
            plt.legend(loc='best')
        except Exception:
            pass
    plt.savefig(dir + '/CDF_cones.pdf')

    full_lcs = []
    for i in range(len(reps)):
        full_lcs.append(int(min(num_qubits[i], 2 * np.floor(k_list[i] / 2) + 2 * reps[i])))

    # plot results
    palette = sns.color_palette("colorblind")


    plt.figure(figsize=(30, 20))
    print(variance_loss)
    if plot_variance:
        plt.plot(num_qubits, variance_loss, 'o-', label=r'Var$\left[\mathcal{L}\right]$', color=palette[1])
    plt.plot(num_qubits, upper_bound, 'o-', label='upper bound', color=palette[0])
    plt.fill_between(num_qubits, np.subtract(upper_bound, est_error_upper_bound), np.add(upper_bound,
                                                                                         est_error_upper_bound),
                     color=palette[0], alpha=.1)
    plt.plot(num_qubits, lower_bound, 'o-', label='lower bound', color=palette[2])
    plt.fill_between(num_qubits, np.subtract(lower_bound, est_error_lower_bound), np.add(lower_bound,
                                                                                         est_error_lower_bound),
                     color=palette[2], alpha=.1)
    plt.ylabel('Variance / bounds')
    plt.xlabel('Number of qubits')
    plt.xticks(num_qubits)
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(dir + '/loss_bounds.pdf')