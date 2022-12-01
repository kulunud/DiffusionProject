import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils2 import *
from modules32 import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, sigma_min=1e-4, sigma_max=1, img_size=128, device="cuda", p = 1.0):
        self.p = p
        self.noise_steps = noise_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.img_size = img_size
        self.device = device
        
        #self.beta = self.prepare_noise_schedule().to(device)
        #self.alpha = 1. - self.beta
        #self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        #prepare noise schedule:
        schedule = self.prepare_noise_schedule()
        self.sigma = schedule[0]
        self.sigma = self.sigma.to(device)
        #print(self.sigma)
        self.s = schedule[1]
        self.s = self.s.to(device)
        #print(self.s)
        self.sigmagrad = schedule[2]
        self.sigmagrad = self.sigmagrad.to(device)
        self.sgrad = schedule[3]
        self.sgrad = self.sgrad.to(device)
        
### change this
    def prepare_noise_schedule(self):
        p = self.p
        i = torch.arange(1, self.noise_steps)/(self.noise_steps-1)
        #reverse time
        i = torch.flip(i, (0,))
        print(i)
        t = (self.sigma_max**(1.0/p) + i*(self.sigma_min**(1.0/p) - self.sigma_max**(1.0/p)))**p
        sigmagrad = (1.0/(self.noise_steps-1.0))*(self.sigma_min**(1.0/p) - self.sigma_max**(1.0/p))*p*(self.sigma_max**(1.0/p) + i*(self.sigma_min**(1.0/p) - self.sigma_max**(1.0/p)))**(p-1.0)
        schedule = torch.stack((t, torch.ones(self.noise_steps-1), sigmagrad, torch.zeros(self.noise_steps-1)))
        print(schedule)
        return schedule

    ### change this - adds noise for each training step forward noising
    def noise_images(self, x, t):      
        #s_noise = self.s[t][:, None, None, None]
        #print(s_noise)
        sigma_noise = self.sigma[t][:, None, None, None]
        Ɛ = torch.randn_like(x)
        return  x + sigma_noise * Ɛ, Ɛ

    ### change this
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps-1, size=(n,))

    ### change this to ODE solve (Euler method)
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x =self.sigma_max*torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            #sigma_t0*s_t0*torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            #time_steps = self.sigma  #dont want this to be linear also dont include zero
            prev_step = self.sigma_max
            for i in tqdm(reversed(range(1, self.noise_steps-1)), position=0):  ## noise steps are time steps...
                t = (torch.ones(n)*i).long().to(self.device)   
                              
                predicted_noise = model(x, t)  #this is D_theta
                sigmagrad =  self.sigmagrad[t][:, None, None, None]
                sigma = self.sigma[t][:, None, None, None]
                sgrad = self.sgrad[t][:, None, None, None]
                s = self.s[t][:, None, None, None]
                                
                di = (sigmagrad/sigma + sgrad/s)*x - (sigmagrad*s/sigma)*predicted_noise
                x = x + (self.sigma[i]-prev_step)*di
                prev_step = self.sigma[i]
                
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args, dataloader):
    setup_logging(args.run_name)
    device = args.device
    dataloader = dataloader
    p = args.p ##need to implement
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device, p = args.p)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)  #whats this doing
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            #print(predicted_noise)
            loss = mse(noise, predicted_noise)
            #print(loss)

            optimizer.zero_grad()
            loss.backward()
            #print(model.encoder.layer1[1].weight.grad)
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 32
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
