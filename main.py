import scanpy as sc
import torch

from torch.utils.data import DataLoader
import torch.autograd as autograd
from torch.autograd import Variable

from model import Generator,Discriminator
from dataset import ScDataset

from utils import setup_seed
from utils import weights_init_normal

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def calculate_gradient_penalty(real_data, fake_data, D):
    eta = torch.FloatTensor(real_data.size(0),1).uniform_(0,1)
    eta = eta.expand(real_data.size(0), real_data.size(1))
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        eta = eta.cuda()
    else:
        eta = eta

    interpolated = eta * real_data + ((1 - eta) * fake_data)

    if cuda:
        interpolated = interpolated.cuda()
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda() if cuda else torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty


def train(scd, n_epochs, batch_size, n_critic):
    data_size = scd.ppd_adata[0].shape[1]
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    latent_dim = 256


    dataloader = DataLoader(
        dataset = scd,
        batch_size = batch_size,
        shuffle = True
    )

    # Initialize generator and discriminator
    G_AB = Generator(data_size,latent_dim)
    D_B = Discriminator(data_size)
    mse_loss = torch.nn.MSELoss()
    if cuda:
        G_AB.cuda()
        D_B.cuda()
        mse_loss.cuda()
        
    # Initialize weights
    G_AB.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    optimizer_G_AB = torch.optim.Adam(G_AB.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        G_AB.train()
        for i, (data_A, data_B) in enumerate(dataloader):
            
            real_data = Variable(data_B.type(FloatTensor))
            optimizer_G_AB.zero_grad()
            z = Variable(data_A.type(FloatTensor), requires_grad=True)
            gen_data = G_AB(z)
            ae_loss = mse_loss(gen_data,real_data) * data_A.size(-1)
            ae_loss.backward()
            optimizer_G_AB.step()
            
            
            # Configure input
            real_data = Variable(data_B.type(FloatTensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D_B.zero_grad()
            z = Variable(data_A.type(FloatTensor))
            gen_data = G_AB(z)


            # Loss for real images
            real_validity  = D_B(real_data)
            fake_validity  = D_B(gen_data)


            # Compute W-div gradient penalty
            div_gp = calculate_gradient_penalty(real_data, gen_data, D_B)

            # Adversarial loss
            db_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10*div_gp
            db_loss.backward()
            optimizer_D_B.step()


            # -----------------
            #  Train Generator
            # -----------------

            if i % n_critic == 0:
                optimizer_G_AB.zero_grad()
                real_data = Variable(data_B.type(FloatTensor))
                z = Variable(data_A.type(FloatTensor), requires_grad=True)
                gen_data = G_AB(z)
                fake_validity = D_B(gen_data)
                gab_loss = -torch.mean(fake_validity)
                gab_loss.backward()

                optimizer_G_AB.step()


        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f] [AE loss: %f]"
            % (epoch+1, n_epochs,
               db_loss.item(),
               gab_loss.item(),
               ae_loss.item()
              )
        )


    G_AB.eval()
    with torch.no_grad():
        z = Variable(FloatTensor(scd.all_adata.X))
        static_sample = G_AB(z)
        fake_data = static_sample.cpu().detach().numpy()
    return fake_data


def integrate_data(ppd_adata, all_adata, batch_key='batch', n_epochs=150, mnn_times=5, len_weight=5, self_nbs=10, other_nbs=1, batch_size=1024, n_critic=100, seed=8,overlap=True,sample_method='max',batch_num=2,exclude_list=[],under_sample=False,under_sample_num=20000,balance_sampling=False):
    if seed is not None:
        setup_seed(seed)
    
    scd = ScDataset(ppd_adata, all_adata, batch_key, mnn_times, len_weight, self_nbs, other_nbs,overlap,sample_method,batch_num,exclude_list,under_sample,under_sample_num,balance_sampling)
    res_data = train(scd,n_epochs,batch_size,n_critic)

    return res_data
