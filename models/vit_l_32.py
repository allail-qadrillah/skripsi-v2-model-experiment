import torch
import timm

def vit_l_32(weight_path: str, num_classes: int, device: torch.device = torch.device("cpu")):
    """
    Load and return a pre-trained ViT-B/32 model.

    Parameters:
        weight_path (str): Path to the pre-trained weight file.

    Returns:
        torch.nn.Module: Pre-trained ViT-B/32 model with loaded weights.
    """
    # setup ViT model instance with pretrained weight
    pretrained_vit = timm.create_model(
        "vit_large_patch32_224", pretrained=False, num_classes=num_classes)

    # load pretrained weights
    pretrained_vit.load_state_dict(
        torch.load(weight_path, map_location=device))

    # Freeze all layers
    for param in pretrained_vit.parameters():
        param.requires_grad = False

    # Unfreeze the final linear layer (head)
    for param in pretrained_vit.head.parameters():
        param.requires_grad = True

    return pretrained_vit
