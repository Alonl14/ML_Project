import sys, os, math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import QuantileTransformer as qt
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import Generator
from torch.nn.functional import kl_div as kld
from torch.nn.functional import log_softmax
from Discriminator import Discriminator
from ParticleDataset import ParticleDataset

QT = qt(output_distribution='normal',n_quantiles=20000,subsample=100000)
#data_path = '/Users/alonlevi/PycharmProjects/ML_Project/Full_Sim_10M.csv' ### this is for local running
data_path = '/storage/agrp/alonle/GAN_InputSample/Full_Sim_10M.csv' ### this is from WIS cluster
#norm_path = '/Users/alonlevi/PycharmProjects/ML_Project/Full_Sim_55M_stats.csv'### this is for local running
norm_path = '/storage/agrp/alonle/GAN_InputSample/Full_Sim_55M_stats.csv'
#output_dir = '/Users/alonlevi/PycharmProjects/ML_Project/saved_Gen.pt' ### this is for local running
output_dir = '/storage/agrp/alonle/GAN_Output/saved_Gen.pt'

dataset = ParticleDataset(data_path, norm_path,QT)
dataloader = DataLoader(dataset.data, batch_size = 2**9, shuffle = True)

N_z=100
print(torch.cuda.is_available())
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    mps_device = torch.device('cuda')
else:
    mps_device = torch.device('mps')
    print('mps')

net_G = Generator.Generator(N_z).to(mps_device)
net_D = Discriminator().to(mps_device)
optimizer_G = optim.Adam(net_G.parameters(), lr = 0.0001, betas= (0.5,0.999))
optimizer_D = optim.Adam(net_D.parameters(), lr = 0.0001, betas= (0.5,0.999))


def get_gradient(crit, real, fake,
                 epsilon):

    mixed_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_images)

    gradient = torch.autograd.grad(
        inputs = mixed_images,
        outputs = mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


G_losses = []
D_losses = []

num_epochs = sys.argv[1]
Lambda = 10 ** (-3)
n_crit = 5

generated_df = pd.DataFrame([])
KLD = torch.tensor([], device=mps_device)
print("Starting Training Loop...")
num_epochs = np.int64(num_epochs)
for epoch in tqdm.tqdm_notebook(range(num_epochs), desc=' epochs', position=0):
    net_G.to(mps_device)
    net_G.train()
    iters = 0
    S = torch.tensor([0.0],device=mps_device)

    avg_error_G, avg_error_D = 0, 0
    avg_Dx, avg_DGz1, avg_DGz2 = 0., 0., 0.

    for i, data in tqdm.tqdm_notebook(enumerate(dataloader, 0), desc=' batch', position=1, leave=False):

        # Update the discriminator network

        ## Train with all-real batch
        for crit_train in range(n_crit):
            net_D.zero_grad()
            b_size = len(data)
            real_data = data.to(mps_device)

            output = net_D(real_data)

            err_D_real = -torch.mean(output)
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, N_z, device=mps_device)
            fake_p = net_G(noise)

            output = net_D(fake_p.detach())
            err_D_fake = torch.mean(output)
            fake_p.to(mps_device)

            epsilon = torch.rand(1, device=mps_device, requires_grad=True)
            gradient = get_gradient(net_D, real_data, fake_p.detach(), epsilon)
            gradient_norm = gradient.norm(2, dim=1)
            penalty = Lambda * torch.mean(torch.square(gradient_norm - 1))

            err_D = err_D_real + err_D_fake + penalty
            err_D.backward()

            # update the discriminator network
            optimizer_D.step()

        # Update the Generator network
        net_G.zero_grad()
        output = net_D(fake_p)
        err_G = -torch.mean(output)
        err_G.backward()

        # update the generator network
        optimizer_G.step()

        # computing the average losses and discriminator
        avg_error_G += err_G.item()
        avg_error_D += err_D.item()

        S += torch.tensor([kld(log_softmax(fake_p, dim=1), log_softmax(real_data, dim=1), log_target=True)],
                          device=mps_device)

        iters += 1

    torch.save(net_G.state_dict(), output_dir)

    KLD = torch.cat((KLD, S/iters), dim=0)
    
    G_losses.append(avg_error_G / iters)
    D_losses.append(avg_error_D / iters)
    print(f'{epoch}/{num_epochs}\tLoss: {(avg_error_D + avg_error_G) / iters:.4f}')

