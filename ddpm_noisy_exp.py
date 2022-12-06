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
import pytorch_gan_metrics
from pytorch_gan_metrics import get_inception_score, get_fid

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.05, img_size=32, device="cuda", p = 1.0):
        self.p = p
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        print(self.beta_end)
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        #print(torch.sqrt(1 - self.alpha_hat))
    def prepare_noise_schedule(self):
        p = self.p
        t = torch.arange(10**(-5), 1.0, 1.0/self.noise_steps)
        beta = self.beta_start*(self.beta_end/self.beta_start)**t
        return beta

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def BuildX(p, model, diffusion, n):
  x = torch.empty((0,3, 32, 32)).to('cuda')
  for k in range(10):
      z = diffusion.sample(model, n)
      x = torch.cat((x, z))
  return x


def train(args, dataloader):
    setup_logging(args.run_name)
    device = args.device
    dataloader = dataloader
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(beta_end=args.beta_end, img_size=args.image_size, device=device, p = args.p)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    FID_vec = torch.empty((1)).to(device)
    for epoch in range(args.epochs):
        print(epoch)
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        #sampled_images = diffusion.sample(model, n=images.shape[0])
        #save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        
        if epoch==49 or epoch==99 or epoch==499 or epoch==999:
            sampled_x = BuildX(p = args.p, model = model, diffusion = diffusion, n = args.n)
            image_tensor = sampled_x/255
            FID = get_fid(image_tensor, '/content/DiffusionProject/data/cifar10.train.npz')
            print(FID)
            FID_vec = torch.cat((FID_vec, torch.Tensor([FID]).to(device))).to(device)
        
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
                            
    print(FID_vec)
    torch.save(FID_vec, os.path.join("models", args.run_name, f"FIDScores.pt"))
                         


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
