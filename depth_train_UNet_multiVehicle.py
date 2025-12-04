import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#import pandas as pd
import imageio.v2 as imageio
import argparse
import matplotlib.pyplot as plt




H = 480
W = 640

MAX_DEPTH_METERS = 300.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def decodeCarlaDepth(path):
    # helpder function to decode

    # have to decode png into int8
    raw = io.read_image(path)  
    raw = raw.permute(1,2,0).numpy().astype(np.float32) 

    R = raw[:,:,0]
    G = raw[:,:,1]
    B = raw[:,:,2]

    normalized = (R + 256.0 *G + 65536.0 *B) / (256**3 - 1)

    depth_m = normalized * 1000.0
    # should be shape (H, W)

    # this clamps in case we did some weird nan thing
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=MAX_DEPTH_METERS, neginf=0.0)
    depth_m = np.clip(depth_m, 0.0, MAX_DEPTH_METERS)

    return depth_m 



# this version reads for multiple vehicles
class MultiAgentDepthDataset(Dataset):
    def __init__(self, root_dir, h, w):

        self.h = h
        self.w = w

        self.samples = []

        agent_dirs = [os.path.join(root_dir, d) 
                      for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]

        for agent_path in agent_dirs:
            rgb_path = os.path.join(agent_path, "rgb")
            depth_path = os.path.join(agent_path, "depth_raw")

            if not (os.path.isdir(rgb_path) and os.path.isdir(depth_path)):
                print(f"[WARN] Missing rgb/depth in {agent_path}, skipping")
                continue

            rgb_files = sorted(os.listdir(rgb_path))

            for f in rgb_files:
                rgb_file = os.path.join(rgb_path, f)
                depth_file = os.path.join(depth_path, f)

                if os.path.exists(depth_file):
                    self.samples.append((rgb_file, depth_file))
                else:
                    print(f"[WARN] Missing mask for {rgb_file}, skipping")

        print(f"[INFO] Loaded {len(self.samples)} (rgb, depth) pairs from {len(agent_dirs)} agents.")

        # transforms
        self.image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3])  # keep RGB only
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_file, depth_file = self.samples[index]

        # rgb loading
        img = io.read_file(rgb_file)
        img = io.decode_png(img)
        img = self.image_transforms(img)

        # depth loading
        # (H,W)
        depth_m = decodeCarlaDepth(depth_file)  
        # new shape should be (1,H,W)
        depth_m = torch.from_numpy(depth_m).unsqueeze(0)  
        depth_m = self.mask_transforms(depth_m)  
        # resized
        return {"IMAGE": img, "DEPTH": depth_m}
    

# U-net encoder consists of several convolutional blocks that typically have the same layout:
# kernel size 3x3 with 1 padding to maintain input size with each filter
# reLU activation
# maxpool of kernel size 2x2 stride 2, to decrease resolution by 0.5 in both dimensions
# dropout
# i added batchnorm just in case we want to try that
class convStack(nn.Module):
    def __init__(self, in_channels, n_filters=32, p_dropout=0, maxpool=True, batchNorm=False):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, n_filters, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters) # batchnorm occurs after convolution
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters) # here too
        self.activation = nn.ReLU()
        self.maxpool = maxpool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if self.maxpool else None
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(p=p_dropout)
        self.batchNorm = batchNorm


    def forward(self, x):
            # forward through the convolutional block
            out = self.conv_1(x)
            # do batchnorm if set to true
            if self.batchNorm:
                out = self.bn1(out)
            out = self.activation(out)
            out = self.conv_2(out)
            # do batchnorm if set to true
            if self.batchNorm:
                out = self.bn2(out)
            out = self.activation(out)
            # apply dropout if set
            if self.p_dropout > 0:
                out = self.dropout(out)

            # save the skip connection here before maxpooling (remember we wanna keep it the original size)
            skip_connection = out.clone()
            # ok now we can maxpool
            if self.maxpool:
                out = self.pool(out)
            next_layer = out
            # return both 
            return next_layer, skip_connection



# now we need the decoder
# similar to encoder block (kernel sizes, sampling rates) except we usually dont use dropout here since we want all the info we can obtain
# we upsample, then we add the skip inputs from the other side of the "U", and filter again
# then activation
# decoder usually doesnt have batchnorm because..
# we're not searching for the features anymore, we're building the mask
# added it in just in case i wanna experiment with it though
class decoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, n_filters=32, batchNorm=False):
        super().__init__()
        # the upsampler
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=(2, 2), stride=2, padding=0)
        # heres where the skip connection is added in
        self.conv_1 = nn.Conv2d(in_channels // 2 + skip_channels, n_filters, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.activation = nn.ReLU()
        self.batchNorm = batchNorm
    
    def forward(self, input, skip_input):

        out = self.upsample(input)
        #where we concatenate the skip connection
        out = torch.cat([out, skip_input], dim=1) 
        out = self.conv_1(out)

        if self.batchNorm:
                out = self.bn1(out)

        out = self.activation(out)
        out = self.conv_2(out)

        if self.batchNorm:
                out = self.bn2(out)

        out = self.activation(out)
        return out



# U-Net in action
# as seen in class its a series of convolutional blocks/stacks followed by a series of upsampling
# the number of filters increases alot towards the middle of the network
# 
class CARLA_UNet(nn.Module):
    def __init__(self, in_channels=3, n_filters=16, n_classes=28, batchNorm=False):
     
        # notice dropout is usually done in the denser middle part where we have lots of filters and low level features
        # not a good idea to do it in other places or in upsampling
        super().__init__()
        self.conv_layer_1 = convStack(in_channels, n_filters, batchNorm=batchNorm)
        self.conv_layer_2 = convStack(n_filters, n_filters*2, batchNorm=batchNorm)
        self.conv_layer_3 = convStack(n_filters*2,  n_filters*4, batchNorm=batchNorm)
        self.conv_layer_4 = convStack(n_filters*4, n_filters*8, p_dropout=0.3, batchNorm=batchNorm)
        # final encoder block doesnt maxpool again because now we're going back up in resolution
        self.conv_layer_5 = convStack(n_filters*8, n_filters*16, p_dropout=0.3, maxpool=False, batchNorm=batchNorm)

        self.upsample_1 = decoderBlock(n_filters*16, n_filters*8, n_filters * 8, batchNorm=batchNorm)
        self.upsample_2 = decoderBlock(n_filters*8, n_filters*4, n_filters * 4, batchNorm=batchNorm)
        self.upsample_3 = decoderBlock(n_filters*4, n_filters*2, n_filters * 2, batchNorm=batchNorm)
        self.upsample_4 = decoderBlock(n_filters*2, n_filters*1, n_filters * 1, batchNorm=batchNorm)

        # and then one last set of sequential for good measure
        self.last_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters,  kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            #nn.Conv2d(n_filters, n_classes,  kernel_size=(1, 1), padding=0),
            # this outputs a single depth channel
            nn.Conv2d(n_filters, 1, kernel_size=1) 
        )

    def forward(self, x):
   
        # see now the skip connection is reintroduced at the proper resolution
        # input resolution
        conv_1_next, conv_1_skip = self.conv_layer_1(x)
        # after one maxpol
        conv_2_next, conv_2_skip = self.conv_layer_2(conv_1_next)
        # after two maxpools
        conv_3_next, conv_3_skip = self.conv_layer_3(conv_2_next)
        # after three maxpools
        conv_4_next, conv_4_skip = self.conv_layer_4(conv_3_next)
        # after four maxpools
        conv_5_next, conv_5_skip = self.conv_layer_5(conv_4_next)
        

        # combine the output with the previous skip corresponding to the right resolution
        out = self.upsample_1(conv_5_next, conv_4_skip)
        # and again
        out = self.upsample_2(out, conv_3_skip)
        # and again
        out = self.upsample_3(out, conv_2_skip)
        # and again
        out = self.upsample_4(out, conv_1_skip)
        # and one final convolution for good measure
        out = self.last_conv(out)
        return out
    

# combines L1 and L2
class BerHuLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        diff = torch.abs(target - pred)
        c = 0.2 * diff.max()

        # L1 
        l1_part = diff[diff <= c]

        # L2 
        l2_part = diff[diff > c]
        l2_part = (l2_part**2 + c**2) / (2*c)

        loss = torch.cat([l1_part, l2_part], dim=0).mean()
        return loss
        

def depth_metrics(pred, target):

    # returns dictionary of metrics
    eps = 1e-6
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    diff = pred - target
    # mae,
    mae = diff.abs().mean().item()
    # rmse, 
    rmse = torch.sqrt((diff ** 2).mean()).item()

    # absrel, 
    absrel = (diff.abs() / target).mean().item()

    # delta1/2/3 (averaged per batch)
    ratio = torch.max(pred / target, target / pred)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {"mae": mae, "rmse": rmse, "absrel": absrel, "delta1": delta1, "delta2": delta2, "delta3": delta3}

# som hyperparams

# EPOCHS = 30
# BATCH_SIZE = 32
# LR = 0.001
# B1 = 0.9
# B2 = 0.999



# main training function

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("path", help="folder containing the multiple agent data")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--b1", type=float, default=0.9, help="Adam beta1") # im probably not gonna mess with these
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2") # this neither

    args = parser.parse_args()

    #path = 'output'
    #image_path = os.path.join(args.path, 'rgb')
    #print(image_path)
 

    # load_images()
    # set up the dataloaders

    root_dir = "output"   # we are doing multi now!!
    #root = "output"

    # count = 0

    # for path, dirs, files in os.walk(root_dir):
    #     for f in files:
    #         if f.lower().endswith(".png"):
    #             full = os.path.join(path, f)
    #             count += 1

    #             # progress log every 250 images
    #             if count % 250 == 0:
    #                 print(f"Checked {count} PNG files so far...")

    #             try:
    #                 io.decode_png(io.read_file(full))
    #             except Exception as e:
    #                 print("CORRUPTED:", full, e)

    dataset = MultiAgentDepthDataset(root_dir, H, W)
    # 20% validation
    val_ratio = 0.2  
    n_total = len(dataset)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_set, val_set = torch.utils.data.random_split(
        dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)  # reproducible
    )
    train_loader = DataLoader(train_set, batch_size=args.batchSize, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=args.batchSize, shuffle=False)
    # print(len(dataloader))
    # now set up the model
    #unet = CARLA_UNet().to(device)
    depth_unet = CARLA_UNet(in_channels=3, n_filters=16, n_classes=1).to(device)
    print(summary(depth_unet, (3, H, W)))
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    #criterion = BerHuLoss()
    # adam optimizer
    optimizer = torch.optim.Adam(depth_unet.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # step scheduler decreases learning rate every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,   
        gamma=0.5       
    )
    
    # training loop
    os.makedirs("unet_depth", exist_ok=True)
    os.makedirs("depth_plots", exist_ok=True)

    #USE_AMP = False  # set False to temporarily disable mixed precision for debugging
    #scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and USE_AMP))

    train_losses = []
    val_losses = []

    save_counter = 0

    best_val_rmse = float('inf')

    for epoch in range(args.epochs):
        depth_unet.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            images = batch["IMAGE"].to(device, non_blocking=True)
            depths = batch["DEPTH"].to(device, non_blocking=True)

            optimizer.zero_grad()
            #with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            
            
            outputs =  depth_unet(images)  # (B,1,H,W)
            outputs = torch.clamp(outputs, 0.0, MAX_DEPTH_METERS)

            loss = criterion(outputs, depths)
            
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()


            
            running_loss += loss.item() * images.size(0)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(depth_unet.parameters(), max_norm=1.0)
            #torch.nn.utils.clip_grad_value_(depth_unet.parameters(), clip_value=1.0)
            optimizer.step()
            # logging
            # print how many samples its gone thru
            currSamples = min((i + 1) * args.batchSize, len(train_loader.dataset))
            print(f'EPOCH {epoch} ({currSamples}/{len(train_loader.dataset)})  \t Loss:{loss.item()}')

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # validation

        depth_unet.eval()
        val_loss = 0

        metrics_acc = {"mae": 0.0, "rmse": 0.0, "absrel": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
        batches = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["IMAGE"].to(device, non_blocking=True)
                depths = batch["DEPTH"].to(device, non_blocking=True)

                #with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                #    outputs = depth_unet(images)

                # optimizer.zero_grad()
                
                outputs = depth_unet(images)

                outputs = torch.clamp(outputs, 0.0, MAX_DEPTH_METERS)

                loss = criterion(outputs, depths)
                val_running_loss += loss.item() * images.size(0)

                # collect metrics
                m = depth_metrics(outputs, depths)
                for k in metrics_acc:
                    metrics_acc[k] += m[k]
                batches += 1

               
            # average metrics
            val_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            for k in metrics_acc:
                metrics_acc[k] /= max(1, batches)

            print(f"Epoch {epoch}/{args.epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f" MAE: {metrics_acc['mae']:.4f} m | RMSE: {metrics_acc['rmse']:.4f} m | AbsRel: {metrics_acc['absrel']:.4f}")
            print(f" delta1: {metrics_acc['delta1']:.4f} | delta2: {metrics_acc['delta2']:.4f} | delta3: {metrics_acc['delta3']:.4f}")

            # save model checkpoint
            torch.save(depth_unet.state_dict(), f"unet_depth/unet_depth_epoch_{epoch:03d}.pth")
            print(f"Saved 'unet_depth/unet_depth_epoch_{epoch:03d}.pth'")

            # optionally save best by RMSE
            if metrics_acc["rmse"] < best_val_rmse:
                best_val_rmse = metrics_acc["rmse"]
                torch.save(depth_unet.state_dict(), "unet_depth/unet_depth_best.pth")
                print("Saved best model checkpoint.")

        scheduler.step()

    # done training, get some plots and visualizations

    print("Training finished.")

    print("Generating depth validation visualizations...")

    vis_dir = "depth_plots/val_predictions"
    os.makedirs(vis_dir, exist_ok=True)

    depth_unet.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):

            images = batch["IMAGE"].to(device)
            depths = batch["DEPTH"].to(device)

            # forward pass
            #with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            
            
            preds = depth_unet(images)

            # detach to cpu
            images_np = images.cpu().permute(0,2,3,1).numpy()            # (B,H,W,3)
            gt_np     = depths.cpu().squeeze(1).numpy()                  # (B,H,W)
            pred_np   = preds.cpu().squeeze(1).numpy()                   # (B,H,W)

            B = images_np.shape[0]

            for i in range(B):
                rgb     = images_np[i]              # float in [0,1]
                gt_d    = gt_np[i]                  # meters
                pred_d  = pred_np[i]                # meters

                # normalize depth maps for visualization
                vmax = MAX_DEPTH_METERS
                gt_viz   = np.clip(gt_d / vmax, 0, 1)
                pred_viz = np.clip(pred_d / vmax, 0, 1)

                # create fig
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                ax[0].imshow(rgb)
                ax[0].set_title("RGB")
                ax[0].axis("off")

                ax[1].imshow(gt_viz, cmap="magma")
                ax[1].set_title("Ground Truth Depth")
                ax[1].axis("off")

                ax[2].imshow(pred_viz, cmap="magma")
                ax[2].set_title("Predicted Depth")
                ax[2].axis("off")

                fig.tight_layout()
                fig.savefig(os.path.join(vis_dir, f"val_{idx:04d}_{i:02d}.png"), dpi=200)
                plt.close(fig)

    print(f"Saved visualization images to: {vis_dir}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss (meters)")
    plt.title("Depth training loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("depth_plots/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved in depth_plots/loss_curve.png'")

    
    



    


    




















