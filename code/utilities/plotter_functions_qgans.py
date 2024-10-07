from collections.abc import Iterable

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from matplotlib.gridspec import GridSpec
from matplotlib import cm
import seaborn as sns

sns.set_theme(style="white", palette="colorblind")
plt.rc('font', size=30)
plt.rc('axes', titlesize=40, titlepad=32, labelsize=46, labelpad=30)
plt.rc('xtick', labelsize=40)
plt.rc('ytick', labelsize=40)
plt.rc('legend', fontsize=35)
plt.rc('lines', linewidth=6)


def plotter_pdfs(plotting_dist: Iterable = None,
                 plotting_gen_dist: Iterable = None,
                 num_qubits: int = 6,
                 file_dir: str = None):
    """
    Plot discrete probability density functions (pdfs)
    :param plotting_dist: Target pdf
    :param plotting_gen_dist: Generated pdf
    :param num_qubits: Number of system qubits--defines the number of discrete points
    :param file_dir: Directory to store plot to
    :return:
    """

    cmap = cm.Blues  # cmap = cm.cividis
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(1, 2, figure=fig)

    if isinstance(plotting_dist, list):
        print('List  plotting dist')
        temp = []
        for p_dist in plotting_dist:
            X, Y, Z, init_gen_dist, gen_dist = p_dist
            gen = gen_dist.reshape(X.shape).detach().numpy()
            temp.append(gen)
        gen = np.average(temp, axis=0)
    else:
        X, Y, Z, init_gen_dist, gen_dist = plotting_dist
        if plotting_gen_dist is not None:
            gen_dist = plotting_gen_dist
        gen = gen_dist.reshape(X.shape).detach().numpy()

    zticks = [2e-2, 6e-2, 1e-1] if num_qubits == 6 else [2e-5, 6e-5, 1e-4]
    zlim = 1e-1 if num_qubits == 6 else 1e-4
    zticklabels = ["2", "6", "10"]
    xticks = [-1, 0, 1] if num_qubits == 6 else [-3, 0, 3]

    # True Dist
    ax1 = fig.add_subplot(gs[:, 0], projection="3d", zorder=1)
    ax1.plot_surface(X, Y, Z, cmap=cmap, linewidth=0)
    ax1.set_zticks(zticks)
    ax1.set_zlim(0, zlim)
    ax1.set_zticklabels(zticklabels)
    ax1.set_xticks(xticks)
    ax1.set_yticks(xticks)
    ax1.tick_params(axis='z', pad=16)
    ax1.view_init(55, -60)

    # Gen Dist
    ax2 = fig.add_subplot(gs[:, 1], projection="3d")    
    ax2.plot_surface(X, Y, gen, cmap=cmap, linewidth=0)
    ax2.set_zticks(zticks)
    ax2.set_zlim(0, zlim)
    ax2.set_zticklabels(zticklabels)
    ax2.set_xticks(xticks)
    ax2.set_yticks(xticks)
    ax2.tick_params(axis='z', pad=16)
    ax2.view_init(55, -60)
    
    fig.tight_layout()
    plt.savefig("results_" + str(num_qubits) + "_pdfs.pdf", bbox_inches='tight', transparent=True)
    if file_dir==None:
        plt.show()
    else:
        plt.savefig(file_dir)


def plotter_loss(plotting_losses: Iterable,
                 plotting_grads: Iterable = None,
                 num_epochs: int = None,
                 num_qubits: int = 6,
                 k_orders: int = None,
                 std_losses: Iterable = None,
                 std_grads: Iterable = None,
                 file_dir: str = None):
    """
    Plot model loss and gradients over the executed training epochs
    :param plotting_losses: Model loss to be plotted
    :param plotting_grads: Model gradients to be plotted
    :param num_epochs: Number of training epochs
    :param num_qubits: Number of system qubits--defines the number of discrete points
    :param std_losses: Standard deviation of model loss to be plotted
    :param std_grads: Standard deviation of model gradients to be plotted
    :param file_dir: Directory to store plot to
    :return:
    """
    palette = sns.color_palette("colorblind")
    generator_loss_values, discriminator_loss_values, entropy_values = plotting_losses
    if std_losses is not None:
        generator_loss_std, discriminator_loss_std, entropy_std = std_losses
    orders, grads_all, coeffs_all = plotting_grads
    grads = np.linalg.norm(grads_all, axis=-1)
    if std_grads is not None:
        std_grads = np.linalg.norm(std_grads, axis=-1)
    num_epochs = len(generator_loss_values) if num_epochs is None else num_epochs
    fig = plt.figure(figsize=(24, 12))

    ax1 = fig.add_subplot()
    color = palette[0]
    ax1.plot(entropy_values[:num_epochs], color=color)
    if std_losses is not None:
        ax1.fill_between(range(num_epochs), entropy_values[:num_epochs] - entropy_std[:num_epochs],
                         entropy_values[:num_epochs] + entropy_std[:num_epochs], alpha=0.5, color=color)
    ax1.set_ylabel('Relative Entropy', color=color)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["0", "1", "2", "3"])
    ax1.tick_params(axis='y', labelcolor=color)

    colors = [palette[3], palette[1], palette[2], palette[4], palette[5], palette[6]]
    if k_orders is not None:
        ax2 = ax1.twinx()
        ax2.plot(grads[:num_epochs, -1], color=colors[0], label="Full", alpha=0.8)
        for i, order in enumerate(k_orders):
            ax2.plot(grads[:num_epochs, i], label="k = " + str(order), color=palette[np.mod(i, len(palette))],
                     alpha=0.8)
        ax2.legend(loc="best")
        ax2.set_ylabel(r'Full & $k$-Local Gradients', rotation=270, labelpad=60, color=colors[0])
        ax2.tick_params(axis='y', labelcolor=colors[2])
        ax2.set_axisbelow(True)
        ax2.set_zorder(1)
        ax2.set_yscale('log')
    else:
        ax1.set_xlabel('Training Epoch')
    ax1.set_zorder(2)
    ax1.set_axisbelow(True)
    ax1.patch.set_visible(False)

    ax3 = fig.add_subplot()
    ax3.plot(entropy_values[:num_epochs], alpha=0)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(["", "", "", ""])
    ax3.grid()

    fig.tight_layout()
    if file_dir == None:
        plt.show()
    else:
        plt.savefig(file_dir+"/results_" + str(num_qubits) + "_loss.pdf", bbox_inches='tight')
    return

