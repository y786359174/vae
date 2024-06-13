import os
import time
import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
from network import VAE, weights_init_normal
from tqdm import tqdm


import cv2
import einops
import numpy as np

def train(vae:VAE,  ckpt_path, device = 'cuda'):
    
    n_epochs = 10000
    batch_size = 256
    lr = 1e-4
    beta1 = 0.5
    kl_weight = 0.00025 # 重构损失和kl loss的比值
    # kl_weight = 0.001

    recons_fn = nn.MSELoss().to(device)         
    dataloader = get_dataloader(batch_size, num_worker=4)
    optimizer = torch.optim.Adam(vae.parameters(), lr, betas=(beta1, 0.999))    # betas 是一个优化参数，简单来说就是考虑近期的梯度信息的程度
    vae = vae.train()

    for epoch_i in range(n_epochs):
        tic = time.time()
        for x,_ in dataloader:
            if(x.shape[0]!=batch_size):
                continue
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            
            recons_loss = recons_fn(y_hat, x)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)  # 你这sum啊还是mean啊其实随便，加起来就行，我找的code就这么写的，他也把weight调好了我就先不动了
            loss = recons_loss + kl_loss * kl_weight
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        toc = time.time()    
        
        if(epoch_i%20==0):
            vae_weights = {'vae': vae.state_dict()}
            torch.save(vae_weights, ckpt_path)
            sample(vae, device=device)
            print(f'epoch {epoch_i} recons_loss {recons_loss:.4e} kl_loss {kl_loss* kl_weight:.4e} time: {(toc - tic):.2f}s')
        



sample_time = 0
def sample(vae:VAE, device='cuda'):
    global sample_time
    sample_time += 1
    i_n = 5
    # for i in range(i_n*i_n):
    vae = vae.to(device)
    vae = vae.eval()
    with torch.no_grad():
        x_new = vae.sample(i_n*i_n)
        x_new = x_new.detach().cpu()
        x_new = einops.rearrange(x_new, '(n1 n2) c h w -> (n2 h) (n1 w) c', n1 = i_n)
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir,'./vae_sample_%d.jpg' % (sample_time)), x_new)
    vae = vae.train()

save_dir = './data/anime_faces64'
# save_dir = ''

if __name__ == '__main__':

    ckpt_path = os.path.join(save_dir,'model_vae.pth') 
    device = 'cuda'
    image_shape = get_img_shape()
    model = VAE(image_shape[0],image_shape[1]).to(device)

    model.apply(weights_init_normal)

    # gan_weights = torch.load(ckpt_path)
    # gen.load_state_dict(gan_weights['gen'])
    # dis.load_state_dict(gan_weights['dis'])

    train(model, ckpt_path, device=device)
    
    sample(model, device=device)
