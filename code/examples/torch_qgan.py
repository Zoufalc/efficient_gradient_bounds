import time
import pickle
from scipy.stats import entropy
from scipy.sparse import diags

import sys
import os

import numpy as np
import torch

from torch.optim import Adam, RMSprop, SGD

from qiskit.circuit.library import EfficientSU2

from code.utilities.torch_functions import *

from code.utilities.plotter_functions_qgans import *


# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

warm_start = False

# Select true pdf
num_qubits = 4 # Should be a factor of 2

if num_qubits <= 8:
    mus = [[0, 0]]
    sigs = [0.2 for i in range(len(mus))]
else:
    mus = [[s*i, r*j] for i in [1, 3] for j in [1, 3] for r in [-1, 1] for s in [-1, 1]]
    sigs = [0.1 for i in range(len(mus))]

X, Y, Z, grid = generate_gaussian_mixture(mus, sigs, num_qubits, box=np.max(np.abs(mus))+1)
prob_data = Z.reshape(-1)

# Select circuit
odd_layers = [(2*i, 2*i+1) for i in range(num_qubits//2)]
even_layers = [(2*i+1, 2*i+2) for i in range(num_qubits//2-1)]
alternating = [odd_layers, even_layers]

reps = 2
qc = EfficientSU2(num_qubits, reps=reps)

# Select continuous or discrete inputs
continuous_inputs = False
disc_inputs_np = grid if continuous_inputs else unpack(num_qubits).T
disc_inputs = torch.tensor(disc_inputs_np, dtype=torch.float)
gen_inputs = torch.tensor([])

# Select batch sizes
gen_batch_size = int(1e6)
real_batch_size = int(1e6)

# Setup generator + discriminator
generator = Generator(qc, qc.parameters, [], shots=gen_batch_size, seed=seed)
discriminator = Discriminator(disc_inputs.shape[-1], 1)
print("# Generator Parameters =", qc.num_parameters)
print("# Discriminator Parameters =", sum(p.numel() for p in discriminator.parameters() if p.requires_grad))
init_gen_dist = generator(gen_inputs).detach()

# Select optimiser + hyperparameters (no momentum, no weight decay)
gen_opt = 'rms'
gen_lr = 0.01
disc_lr = 0.01
alpha = 0.999
betas = (0.7, 0.999)

if gen_opt == 'rms':
    generator_optimizer = RMSprop(generator.parameters(), lr=gen_lr, alpha=alpha)
elif gen_opt == 'adam':
    generator_optimizer = Adam(generator.parameters(), lr=gen_lr, betas=betas)

# Always select Adam for discriminator (empirically works well)
discriminator_optimizer = Adam(discriminator.parameters(), lr=disc_lr, betas=betas)

# Select Wasserstein or standard GAN
wasserstein = False
wasserstein_clipper = 0.1

# n_epochs = training epochs
n_epochs = 300
# n_critic = discriminator steps per generator step
n_critic = 5

# Select RevGrad or SPSA
# estimator = 'rev'
estimator = 'spsa'

# Should we record gradients? (slower)
plot_gradients = True

if plot_gradients:
    # Select which k-local terms to record
    # orders = [1, 2, 3, num_qubits-2, num_qubits-1, num_qubits]
    orders = [0, 1, 2]
    alphas = get_alphas(num_qubits, orders=orders)
    signs_alphas = (-1)**(alphas@disc_inputs_np.T)
    obs_alphas = build_obs(alphas, sparse=True)
    coeffs_all = []
    grads_all = np.zeros((n_epochs, len(orders)+1, qc.num_parameters))

# Test pickle (before running experiment)
pickle_name = str(num_qubits)+"_"+gen_opt+"_"+estimator+"_"+str(n_critic)+\
              "c_"+str(int(np.log10(gen_batch_size)))+"b_"+str(reps)+"L"
if warm_start == True:
    g_checkpoint = torch.load('g_checkpoint.pth')
    generator_optimizer = g_checkpoint['optimizer']
    d_checkpoint = torch.load('d_checkpoint.pth')
    epoch_start = d_checkpoint['epoch']
    discriminator = d_checkpoint['model']
    discriminator_optimizer = d_checkpoint['optimizer']
    with open(pickle_name, 'rb') as f:
        plotting_losses, plotting_dist, plotting_grads, gen_dist, generator_weights, discriminator_weights = pickle.load(
            f)
        orders, grads_all, coeffs_all = plotting_grads
        generator_loss_values = plotting_losses[0]
        discriminator_loss_values = plotting_losses[1]
        entropy_values = plotting_losses[2]
    generator = Generator(qc, qc.parameters, [], shots=gen_batch_size, seed=seed, initial_weights=generator_weights[-1],)

else:
    generator = Generator(qc, qc.parameters, [], shots=gen_batch_size, seed=seed)
    discriminator = Discriminator(disc_inputs.shape[-1], 1)
    epoch_start = 0
    with open(pickle_name, "wb") as f:
        pickle.dump([], f)
    generator_loss_values, discriminator_loss_values, entropy_values = [], [], []
    gen_dist = []
    generator_weights, discriminator_weights = [], []
    plotting_losses, plotting_dist, plotting_grads = None, None, None

print("# Generator Parameters =", qc.num_parameters)
print("# Discriminator Parameters =", sum(p.numel() for p in discriminator.parameters() if p.requires_grad))
init_gen_dist = generator(gen_inputs).detach()

plot_gradients = True
if warm_start == False:
    if gen_opt == 'rms':
        generator_optimizer = RMSprop(generator.parameters(), lr=gen_lr, alpha=alpha)
    elif gen_opt == 'adam':
        generator_optimizer = Adam(generator.parameters(), lr=gen_lr, betas=betas, amsgrad=True)
    elif gen_opt == 'sgd':
        generator_optimizer = SGD(generator.parameters(), lr=gen_lr)  # momentum, #weight decay, #dampening

    if plot_gradients:
        # Select which k-local terms to record
        # orders = [0] # orders = [1, 2, 3, 4, 5, 6] # oders = [1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        coeffs_all = []
        grads_all = np.zeros((n_epochs, len(orders) + 1, qc.num_parameters))
if plot_gradients:
    alphas = get_alphas(num_qubits, orders=orders)
    # obs_alphas = build_obs(alphas, sparse=True)
    obs_alphas = build_obs_paulis(alphas)
start_time = time.time()
for epoch in range(n_epochs):
    # Train Discriminator
    for _ in range(n_critic):
        # Note : here we compute discriminator output on ALL possible inputs, which is fine for <= 20 qubits,
        # but we should compute discriminator only on real / generated batch if we want to scale higher.
        disc_value = discriminator(disc_inputs)
        
        # Get batches
        gen_batch_dist = generator(gen_inputs).reshape(-1, 1)
        real_batch = np.random.choice(np.arange(len(prob_data)), size=real_batch_size, p=prob_data)
        real_batch_dist = torch.tensor(np.bincount(real_batch, minlength=len(prob_data)) /
                                       real_batch_size).reshape(-1, 1)

        # Compute discriminator loss + gradient + update
        discriminator_optimizer.zero_grad()

        real_loss = f_loss(real_batch_dist, disc_value, wasserstein)
        fake_loss = f_tilde_loss(gen_batch_dist.detach(), disc_value, wasserstein)
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()
                
    # Train generator (get batch + compute loss)
    gen_batch_dist = generator(gen_inputs).reshape(-1, 1)
    disc_value = discriminator(disc_inputs).detach()
    generator_loss = f_loss(gen_batch_dist, disc_value, wasserstein)

    # Compute generator gradient
    generator_optimizer.zero_grad()

    if plot_gradients or estimator == 'rev':
        weights = generator.weight.detach().numpy()
        D_x = -torch.log(torch.sigmoid(disc_value)).numpy().reshape(-1)
    if estimator == 'rev':
        sp_matrix = diags(D_x)
        generator_estimator = Generator_Estimator(qc, [sp_matrix])
        full_grads = generator_estimator.backward(input_data=[], weights=weights)[1][0][0]
    if estimator == 'spsa':  # SPSA
        generator_loss.backward()
        full_grads = generator.weight.grad.numpy()
    generator.weight.grad = torch.FloatTensor(full_grads)

    # Update generator
    generator_optimizer.step()

    # Record losses + entropy
    generator_loss_values.append(generator_loss.detach().item())
    discriminator_loss_values.append(discriminator_loss.detach().item())
    entropy_values.append(entropy(gen_batch_dist.detach().numpy().reshape(-1), prob_data))
    generator_weights.append(weights.tolist())
    discriminator_weights.append(discriminator.state_dict())
    gen_dist.append(gen_batch_dist.reshape(-1, 1))

    # Record gradients (computed using ReverseGradient)
    if plot_gradients:
        coeffs = signs_alphas @ D_x / 2**num_qubits
        try:
            coeffs_orders = [coeffs[np.sum(alphas, axis=1) == order] for order in orders]
        except Exception:
            pass
        obs_weighted = [coeffs[i] * obs_alphas[i] for i in range(len(obs_alphas))]
        obs_orders = [sum([obs_weighted[i] for i in np.flatnonzero(np.sum(alphas, axis=1) == order)]) for order in orders]
        generator_sub = Generator_Estimator(qc, obs_orders)
        grads_orders = generator_sub.backward(input_data=[], weights=weights)[1][0]
        grads_all[epoch, :-1] = grads_orders
        grads_all[epoch, -1] = full_grads
        coeffs_all.append(coeffs_orders)
        plotting_grads = (orders, grads_all, coeffs_all)

    if epoch % 10 == 0: # Print runtime + pickle
        runtime = time.time()-start_time
        print("Epoch: " + str(epoch) + ". Runtime: " + str(int(runtime)) + " seconds.")
        plotting_losses = [generator_loss_values, discriminator_loss_values, entropy_values]
        plotting_dist = (X, Y, Z, init_gen_dist, gen_batch_dist.detach())
        with open(pickle_name, "wb") as f:
            pickle.dump(
                [plotting_losses, plotting_dist, plotting_grads, gen_dist, generator_weights, discriminator_weights], f)

# Print runtime
runtime = time.time()-start_time
print("Total Runtime: " + str(int(runtime//60)) + " minutes.")
print("Total Runtime: " + str(int(runtime//3600)) + " hours.")

# Pickle results
with open(pickle_name, "wb") as f:
    pickle.dump([plotting_losses, plotting_dist, plotting_grads, gen_dist, generator_weights,
                 discriminator_weights], f)
# Print results
plotter_loss(plotting_losses, plotting_grads, n_epochs, num_qubits)
