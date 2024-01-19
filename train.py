import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from model import CAModel
from dataset import CustomDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def dice_loss(input, target):
    smooth = 1.0
    batch_size = input.size(0)
    
    input_flat = input.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    
    intersection = (input_flat * target_flat).sum(1)
    union = input_flat.sum(1) + target_flat.sum(1)
    
    dice = 1 - ((2. * intersection + smooth) / (union + smooth))
    return dice


def iou_loss(preds, target, smooth=1e-6):
    """
    IoU loss vypočtená přímo z plovoucích čísel.

    Args:
    preds (torch.Tensor): Predikce modelu, pravděpodobnosti pro každý pixel.
    target (torch.Tensor): Skutečné masky (ground truth), 0 nebo 1 pro každý pixel.
    smooth (float): Malé číslo přidané pro zamezení dělení nulou.

    Returns:
    torch.Tensor: Průměrná IoU ztráta.
    """

    # Plochá (flatten) preds a target pro snazší výpočet
    preds_flat = preds.view(preds.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    # Výpočet průniku a sjednocení
    intersection = (preds_flat * target_flat).sum()
    union = (preds_flat + target_flat).sum() - intersection

    print("Inter: " + str(intersection))
    print("Union: " + str(union))

    # Výpočet IoU
    iou = (intersection + smooth) / (union + smooth)

    # Výpočet IoU ztráty
    iou_loss = 1 - iou

    # Průměrná ztráta přes dávku
    return iou_loss



def get_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set device to CUDA
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

def to_rgb(img_rgba):
    print(img_rgba.shape)
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)


def make_seed(size, n_channels, fill_channels):
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    # Fill the specified channels with ones
    x[:, fill_channels:, :, :] = 1
    return x

def plot_batch(pool, gt_masks):
    fig, axs = plt.subplots(2, 8, figsize=(20, 10))  # Vytvoří mřížku 4x2 subplotů

    for i in range(8):
        # Vykreslení pool tensoru
        axs[0, i].imshow(pool[i, 3, :, :].detach().cpu(), cmap='gray')
        axs[0, i].set_title(f'Pool Sample {i}')
        axs[0, i].axis('off')  # Skryje osy

        # Vykreslení gt_masks tensoru
        axs[1, i].imshow(gt_masks[i, 0, :, :].cpu(), cmap='gray')
        axs[1, i].set_title(f'GT Mask Sample {i}')
        axs[1, i].axis('off')  # Skryje osy

    plt.tight_layout()  # Zajistí, aby se sub-plots nepřekrývaly
    plt.show()

def plot_batch_adv(pool, gt_masks):
    fig, axs = plt.subplots(4, 8, figsize=(20, 10))  # Vytvoří mřížku 4x2 subplotů

    for i in range(8):
        # Vykreslení pool tensoru
        axs[0, i].imshow(pool[i, 3, :, :].detach().cpu(), cmap='gray')
        axs[0, i].set_title(f'Pool Sample {i}')
        axs[0, i].axis('off')  # Skryje osy

        # Vykreslení pool tensoru
        axs[2, i].imshow(pool[i, 4, :, :].detach().cpu(), cmap='gray')
        axs[2, i].set_title(f'Pool arbitary Sample {i}')
        axs[2, i].axis('off')  # Skryje osy

        # Vykreslení pool tensoru
        axs[3, i].imshow(pool[i, 5, :, :].detach().cpu(), cmap='gray')
        axs[3, i].set_title(f'Pool arbitary 2 Sample {i}')
        axs[3, i].axis('off')  # Skryje osy

        # Vykreslení gt_masks tensoru
        axs[1, i].imshow(gt_masks[i, 0, :, :].cpu(), cmap='gray')
        axs[1, i].set_title(f'GT Mask Sample {i}')
        axs[1, i].axis('off')  # Skryje osy

    plt.tight_layout()  # Zajistí, aby se sub-plots nepřekrývaly
    plt.show()


def main(argv=None):
    batch_size = 8 
    eval_frequency = 999
    eval_iterations = 150
    n_batches = 1000 # For how many batches the training will be
    n_channels = 6 # of the input
    fill_channels = 4
    logdir = "logs"
    padding = 1 # Padding. The shape after padding is (h + 2 * p, w + 2 * p)."
    pool_size = 32
    size = 30
    loss_function = nn.BCEWithLogitsLoss()


    device = get_device()
    log_path = pathlib.Path(logdir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    # Model and optimizer
    model = CAModel(n_channels=n_channels, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    transform = transforms.Compose([
                transforms.Resize((30, 30))
            ])

    # Vytvoření instance datasetu
    dataset = CustomDataset(images_dir='data/train_images/', 
                            masks_dir='data/train_masks/', 
                            transform=transform)

    # Vytvoření dataloaderu
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True)

    count = 0
    for batch in dataloader:
        count += 1
        print(count)
    
    for batch in dataloader:
        images, masks = batch
        counter = 0
        for target_img_, target_mask_ in zip(images, masks):
            counter += 1
            if counter < 4:
                continue

            #target_img_ = nn.functional.pad(target_img_, (padding, padding, padding, padding), "constant", 0)
            #target_mask_ = nn.functional.pad(target_mask_, (padding, padding, padding, padding), "constant", 0)
            target_img = target_img_.to(device)
            target_mask = target_mask_.to(device)

            
            
            # Počet nových kanálů, které chcete přidat
            additional_channels = 2

            # Vytvoření tensoru s doplňkovými kanály (například plný nul)
            #channels_layer = torch.zeros((additional_channels, 30, 30)).to(device)
            
            channels_layer = torch.randn((additional_channels, 30, 30)).to(device)
            
            #channels_layer = torch.empty((additional_channels, 30, 30))
            #channels_layer = torch.nn.init.kaiming_uniform_(channels_layer, mode='fan_in', nonlinearity='relu').to(device)

            # Konkatenace původního tensoru s novými kanály
            target_img = torch.cat((target_img, channels_layer))
            
            #target_img = target_img.repeat(batch_size, 1, 1, 1)

            writer.add_image("ground truth", target_mask) #to_rgb(target_img_)[0])


            # Pool initialization & target mask cloning
            pool = target_img.clone().repeat(pool_size, 1, 1, 1)
            target_masks = target_mask.repeat(batch_size, 1, 1, 1)

            for it in tqdm(range(n_batches)):
                batch_ixs = np.random.choice(pool_size, batch_size, replace=False).tolist()

                x = pool[batch_ixs]
                for i in range(np.random.randint(2, 10)):
                    x = model(x)

                mean_state_values = x.mean(dim=[0, 2, 3])
                
                for ch_num, state_mean in enumerate(mean_state_values):
                    writer.add_scalar("state {ch_num}", state_mean, it)
                

                if it % 2999 == 0:
                    # Uncomment to see beutiful plots of whats going into the network
                    plot_batch_adv(x, target_masks)

                
                loss_batch = ((target_masks - x[:, 3, ...]) ** 2).mean(dim=[1, 2, 3])
                loss = loss_batch.mean()
                #loss_batch = iou_loss(x[:,3,...], target_masks)
                #loss = loss_batch.mean()
                print(loss)
                
                #loss_batch = dice_loss(x[:, 3, ...], target_masks)
                #loss = loss_batch.mean()

                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar("train/loss", loss, it)

                argmax_batch = loss_batch.argmax().item()
                argmax_pool = batch_ixs[argmax_batch]
                remaining_batch = [i for i in range(batch_size) if i != argmax_batch]
                remaining_pool = [i for i in batch_ixs if i != argmax_pool]

                pool[argmax_pool] = target_img.clone()
                pool[remaining_pool] = x[remaining_batch].detach()

                if it % eval_frequency == 0:
                    x_eval = target_img.clone().unsqueeze(0)  # (1, n_channels, size, size)
                    #print(x_eval[:,4,:,:])
                    #print(x_eval.shape)

                    eval_video = torch.empty(1, eval_iterations, 1, *x_eval.shape[2:])
                    eval_video_rgb = torch.empty(1, eval_iterations, 3, *x_eval.shape[2:])
                    #eval_video_arb = torch.empty(1, eval_iterations, 1, *x_eval.shape[2:])
                    #print("shape of eval_video: " + str(eval_video.shape))
                    #print("shape of x_eval: " + str(x_eval.shape))
                    for it_eval in range(eval_iterations):
                        x_eval = model(x_eval)
                        #x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
                        
                        x_eval_out = x_eval[:, 3:4].detach().cpu()
                        x_eval_out_rgb = x_eval[:, :3].detach().cpu()
                        #x_eval_out_arb = x_eval[:, 4:7].detach().cpu()
                        #print("shape of x_eval_out: " + str(x_eval_out.shape))
                        eval_video[0, it_eval] = x_eval_out.type(torch.float32)
                        eval_video_rgb[0, it_eval] = x_eval_out_rgb
                        #eval_video_arb[0, it_eval] = x_eval_out_arb

                    writer.add_video("eval", eval_video, it, fps=20)
                    writer.add_video("eval_rgb", eval_video_rgb, it, fps=20)
                    #writer.add_video("eval_arb", eval_video_arb, it, fps=20)



if __name__ == "__main__":
    main()


#plt.imshow(target_img[:3,...].permute(1,2,0).cpu())
#plt.show()