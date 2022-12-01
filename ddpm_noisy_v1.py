import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils2 import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, sigma_min=1e-4, sigma_max=0.02, img_size=256, device="cuda", p = 1):
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
        self.sigma = prepare.noise_schedule()[0].to(device)
        self.s = prepare.noise_schedule()[1].to(device)
        self.sigmagrad = prepare.noise_schedule()[2].to(device)
        self.sgrad = prepare.noise_schedule()[3].to(device)
        
### change this
    def prepare_noise_schedule(self):
        p = self.p
        i = range(1, self.noise_steps)/(self.noise_steps-1)
        t = (self.sigma_min^(1/p) + i*(self.sigma_min^(1/p) - self.sigma_max^(1/p)))^p
        sigmagrad = p*(self.sigma_min^(1/p) + i*(self.sigma_min^(1/p) - self.sigma_max^(1/p)))^(p-1)*(self.sigma_min^(1/p) - self.sigma_max^(1/p))
        schedule = [t, ones(self.noise_steps), sigmagrad, 0]
        return schedule
        #return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) #linear between beta start and beta end with #noise steps

    ### change this - adds noise for each training step forward noising
    def noise_images(self, x, t):
        #sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        #sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return self.s[t]*x + self.s[t]*self.sigma[t]*Ɛ, Ɛ
        #return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    ### change this
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    ### change this to ODE solve (Euler method)
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = sigma_t0*s_t0*torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            time_steps = self.sigma  #dont want this to be linear also dont include zero
            prev_step = 0
            for i in time_steps: #tqdm(reversed(range(1, self.noise_steps)), position=0):  ## noise steps are time steps...
                t = (torch.ones(n)*i).long().to(self.device)   
                #t = (torch.ones(n) * i).long().to(self.device)
                
                predicted_noise = model(x, t)  #this is D_theta
                
                #alpha = self.alpha[t][:, None, None, None]
                #alpha_hat = self.alpha_hat[t][:, None, None, None]
                #beta = self.beta[t][:, None, None, None]
                #if i > 1:
                #    noise = torch.randn_like(x)
                #else:
                #    noise = torch.zeros_like(x)
                
                di = (self.sigmagrad[t]/self.sigma[t] + self.sgrad[t]/self.s[t])*x - (self.sigmagrad[t]*self.s[t]/self.sigma[t])*predicted_noise
                x = x + (i-prev_step)*di
                prev_step = i
                #x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
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
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
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
    args.image_size = 64
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
