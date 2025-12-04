import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
#import pandas as pd
import imageio.v2 as imageio

import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


H = 480
W = 640

CARLA_PALETTE = [
    (0, 0, 0),          # 0  None
    (70, 70, 70),       # 1  Buildings
    (190, 153, 153),    # 2  Fences
    (250, 170, 160),    # 3  Other
    (220, 20, 60),      # 4  Pedestrians
    (153, 153, 153),    # 5  Poles
    (157, 234, 50),     # 6  RoadLines
    (128, 64, 128),     # 7  Roads
    (244, 35, 232),     # 8  Sidewalks
    (107, 142, 35),     # 9  Vegetation
    (0, 0, 142),        # 10 Vehicles
    (102, 102, 156),    # 11 Walls
    (220, 220, 0),      # 12 TrafficSigns
    (70, 130, 180),     # 13 Sky
    (81, 0, 81),        # 14 Ground
    (150, 100, 100),    # 15 Bridge
    (230, 150, 140),    # 16 RailTrack
    (180, 165, 180),    # 17 GuardRail
    (250, 170, 30),     # 18 TrafficLight
    (110, 190, 160),    # 19 Static
    (170, 120, 50),     # 20 Dynamic
    (45, 60, 150),      # 21 Water
    (145, 170, 100),    # 22 Terrain

    # ---- CARLA extended classes 23–27 ----
    (255, 255, 255),    # 23 Unused / white
    (255, 0, 255),      # 24 Unused / magenta
    (0, 255, 255),      # 25 Unused / cyan
    (255, 255, 0),      # 26 Unused / yellow
    (0, 255, 0),        # 27 Unused / green

]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_images():
#     # get manually collected data
#     path = '/output/'
#     image_path = os.path.join(path, './rgb/')
#     seg_path   = os.path.join(path, './seg/')
#     image_list = os.listdir(image_path)
#     seg_list   = os.listdir(seg_path)
#     image_list = [image_path+i for i in image_list]
#     seg_list = [seg_path+i for i in seg_list]

#     N = 2
#     img = imageio.imread(image_list[N])
#     mask = imageio.imread(seg_list[N])

#     fig, arr = plt.subplots(1, 2, figsize=(14, 10))
#     arr[0].imshow(img)
#     arr[0].set_title('Image')
#     arr[1].imshow(mask[:, :, 0])
#     arr[1].set_title('Segmentation')
#     plt.show()
#     os.makedirs("plots", exist_ok=True)
#     fig.savefig(f"plots/image_{N}.png", dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     return

def colorize_mask(mask, palette=CARLA_PALETTE):
    # mask: HxW torch tensor or numpy array with values 0–27
    # returns: HxWx3 RGB uint8 image
    
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in enumerate(palette):
        color_img[mask == cls_id] = color

    return color_img


# this version reads for multiple vehicles
class MultiAgentSegDataset(Dataset):
    def __init__(self, root_dir, h, w):

        self.h = h
        self.w = w

        self.samples = []

        agent_dirs = [os.path.join(root_dir, d) 
                      for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]

        for agent_path in agent_dirs:
            rgb_path = os.path.join(agent_path, "rgb")
            seg_path = os.path.join(agent_path, "seg_raw")

            if not (os.path.isdir(rgb_path) and os.path.isdir(seg_path)):
                print(f"[WARN] Missing rgb/seg in {agent_path}, skipping")
                continue

            rgb_files = sorted(os.listdir(rgb_path))

            for f in rgb_files:
                rgb_file = os.path.join(rgb_path, f)
                seg_file = os.path.join(seg_path, f)

                if os.path.exists(seg_file):
                    self.samples.append((rgb_file, seg_file))
                else:
                    print(f"[WARN] Missing mask for {rgb_file}, skipping")

        print(f"[INFO] Loaded {len(self.samples)} (rgb, seg) pairs from {len(agent_dirs)} agents.")

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
        rgb_file, seg_file = self.samples[index]

        # load image
        img = io.read_file(rgb_file)
        img = io.decode_png(img)

        # load  mask 
        mask = io.read_image(seg_file)
        
        # converts to 1 channel 
        mask, _ = torch.max(mask[0:3], dim=0, keepdim=True)
        mask = mask.long()

        # apply transforms
        img = self.image_transforms(img)
        mask = self.mask_transforms(mask)
        # print(f"[IDX {index}] mask max pixel value: {mask.max().item()}")
        return {"IMAGE": img, "MASK": mask}
    

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
            nn.Conv2d(n_filters, n_classes,  kernel_size=(1, 1), padding=0),
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
        


# som hyperparams

# EPOCHS = 30
# BATCH_SIZE = 32
# LR = 0.001
# B1 = 0.9
# B2 = 0.999

# stuff for validation
def compute_segmentation_metrics(pred_logits, target_mask, num_classes=28):
    """
    pred_logits: (B, C, H, W)
    target_mask: (B, H, W)
    """
    with torch.no_grad():
        # (B, H, W)
        preds = torch.argmax(pred_logits, dim=1)  

        # Flatten
        preds_flat = preds.view(-1)
        target_flat = target_mask.view(-1)

        # confusion matrix
        conf = torch.bincount(
            num_classes * target_flat + preds_flat,
            minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes).cpu()

        # IoU
        TP = conf.diag()
        FP = conf.sum(dim=0) - TP
        FN = conf.sum(dim=1) - TP
        denom = TP + FP + FN
        # avoid divide by zeros
        IoU = TP.float() / denom.float().clamp(min=1)  

        mIoU = IoU.mean().item()

        pixel_acc = TP.sum().item() / conf.sum().item()

        return {
            "pixel_accuracy": pixel_acc,
            "mIoU": mIoU,
            "IoU_per_class": IoU.tolist()
        }
    

# main training function

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("path", help="folder containing the multiple agent data")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--b1", type=float, default=0.9, help="Adam beta1") # im probably not gonna mess with these
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2") # this neither

    args = parser.parse_args()

    # get paths for rbg images and segmentation masks
    #path = 'output'
    #image_path = os.path.join(args.path, 'rgb')
    #print(image_path)
    #seg_path   = os.path.join(path, 'seg_raw')

    # load_images()
    # set up the dataloaders

    root_dir = "output"   # we are doing multi now!!
    #root = "output"

    count = 0

    # checks if files are corrupted or not usable
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




    dataset = MultiAgentSegDataset(root_dir, H, W)
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
    unet = CARLA_UNet().to(device)
    print(summary(unet, (3, H, W)))
    criterion = nn.CrossEntropyLoss()
    # adam optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # step scheduler decreases learning rate every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,   
        gamma=0.5       
    )
    
    # training loop
    os.makedirs("unet_seg", exist_ok=True)
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        epoch_losses = 0
        unet.train()
        for i, batch in enumerate(train_loader):
            images = batch['IMAGE'].to(device)
            masks = batch['MASK'].to(device)

            N, C, H, W = masks.shape
            masks = masks.reshape((N, H, W)).long()

            optimizer.zero_grad()
            
            outputs = unet(images)
            # print(outputs)
            # compute loss
            loss = criterion(outputs, masks)
            
            epoch_losses += loss.item() * images.size(0)

            # backprop
            loss.backward()
            optimizer.step()


            ## print how many samples its gone thru
            currSamples = min((i + 1) * args.batchSize, len(train_loader.dataset))
            print(f'EPOCH {epoch} ({currSamples}/{len(train_loader.dataset)})  \t Loss:{loss.item()}')
        # do some validation    
        train_loss = (epoch_losses) / len(train_loader.dataset)

        unet.eval()
        val_loss = 0

        # nice confusion matrix
        total_confusion = torch.zeros((28, 28), dtype=torch.long)

        with torch.no_grad():
            for batch in val_loader:
                images = batch['IMAGE'].to(device)
                masks = batch['MASK'].to(device)
                N, C, H, W = masks.shape
                masks = masks.reshape((N, H, W)).long()

                outputs = unet(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)

                # make prediction masks and flatten
                preds = torch.argmax(outputs, dim=1)
                pred_flat = preds.view(-1)
                mask_flat = masks.view(-1)

                # confusion matrix
                conf = torch.bincount(
                    28 * mask_flat + pred_flat,
                    minlength=28 * 28
                ).reshape(28, 28)
                total_confusion += conf.cpu()

        # bunch of metrics
        TP = total_confusion.diag()
        FP = total_confusion.sum(dim=0) - TP
        FN = total_confusion.sum(dim=1) - TP
        IoU = TP.float() / (TP + FP + FN).float().clamp(min=1)
        mIoU = IoU.mean().item()
        pixel_acc = TP.sum().item() / total_confusion.sum().item()


        val_loss /= len(val_loader.dataset)
        print(f"EPOCH {epoch}  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f" Pixel Accuracy: {pixel_acc:.4f}")
        print(f" mIoU: {mIoU:.4f}")
                
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
        # save state dict after every epoch in case of crashes
        torch.save(unet.state_dict(), f"unet_seg/unet_seg_{epoch}.pth")



    # training loop is done, save some graphs

    os.makedirs("seg_plots", exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("seg_plots/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


    print("Generating confusion matrix, be parient")

    # confusion matrix to save
    all_preds = []
    all_labels = []

    vis_dir = "seg_plots/val_predictions"
    os.makedirs(vis_dir, exist_ok=True)
    global_step = 0  # counter for naming images

    unet.eval()

    with torch.no_grad():
        for batch in val_loader:
            images = batch['IMAGE'].to(device)
            masks = batch['MASK'].to(device)

            N, C, H, W = masks.shape
            masks = masks.reshape((N, H, W)).long()

            outputs = unet(images)
            preds = torch.argmax(outputs, dim=1)  

            # save val images
            for b in range(images.size(0)):
                # C,H,W -> H,W,C
                img_rgb = images[b].cpu().permute(1, 2, 0).numpy() 
                img_rgb = (img_rgb * 255).astype(np.uint8)

                # ground truth mask
                gt_mask = masks[b].cpu().numpy().astype(np.uint8)
                gt_rgb = colorize_mask(gt_mask)

                # Predicted mask
                pred_mask = preds[b].cpu().numpy().astype(np.uint8)
                pred_rgb = colorize_mask(pred_mask)

                # add them together
                concat = np.concatenate([img_rgb, gt_rgb, pred_rgb], axis=1)

                # save them all together
                Image.fromarray(concat).save(
                    os.path.join(vis_dir, f"val_{global_step:06d}.png")
                )
                global_step += 1

            # flatten for confusion matrix
            all_labels.append(masks.cpu().numpy().reshape(-1))
            all_preds.append(preds.cpu().numpy().reshape(-1))

    # concatenate all
    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)

    # compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(28)))

    class_labels = [
        "None", "Buildings", "Fences", "Other", "Pedestrians", "Poles", "RoadLines",
        "Roads", "Sidewalks", "Vegetation", "Vehicles", "Walls", "TrafficSigns", "Sky",
        "Ground", "Bridge", "RailTrack", "GuardRail", "TrafficLight", "Static", "Dynamic",
        "Water", "Terrain", "Unused_23", "Unused_24", "Unused_25", "Unused_26", "Unused_27"
    ]

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="viridis", fmt="d",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("CARLA Segmentation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("seg_plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()