import torchvision
import torch

from utils import set_seeds, save_curve, evaluate_model, save_to_json
from models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from data_setup import create_dataloaders
from engine import train
from variables import DIR, BATCH_SIZE, LEARNING_RATE, EPOCH

import os 

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Bismillah Training models Skripsi dengan {device} ...")

# set paths
train_dir = os.path.join(DIR, "train")
val_dir = os.path.join(DIR, "val")
test_dir = os.path.join(DIR, "test")

# set seeds for reproducibility
set_seeds(42)

# define architecture and weights
architecture = {
    "vit_b_16": (vit_b_16, "./weights/vit_base_patch16_224.pth"),
    "vit_b_32": (vit_b_32, "./weights/vit_base_patch32_224.pth"),
    "vit_l_16": (vit_l_16, "./weights/vit_large_patch16_224.pth"),
    "vit_l_32": (vit_l_32, "./weights/vit_large_patch32_224.pth"),
}

if __name__ == "__main__":
    for arch_name, (arch_fn, weight_path) in architecture.items():
        print("="*50)
        print(f"Training {arch_name} ...")
        # define transforms
        if arch_name in ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]:
            pretrained_transform = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()

        # load dataloader
        train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            transforms=pretrained_transform,
            test_dir=test_dir,
            batch_size=BATCH_SIZE
        )
        # setup model
        model = arch_fn(weight_path=weight_path, num_classes=len(class_names), device=device)

        # setup optimizer and loss function
        optimizer = torch.optim.Adam(params=model.parameters(),lr=LEARNING_RATE)
        loss_fn = torch.nn.CrossEntropyLoss()

        # train model
        results = train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=EPOCH,
            device=device
        )
        # save results
        save_to_json(results, f"./results/{arch_name}/results.json")
        
        # save model
        torch.save(model.state_dict(), f"./results/{arch_name}/model.pth")

        # save loss curve
        save_curve(results, arch_name)

        # evaluate model
        evaluate_model(
            model=model,
            dataloader=test_dataloader,
            arch_name=arch_name,
            class_names=class_names,
            device=device
        )

    