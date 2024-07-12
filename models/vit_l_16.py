import torch
import timm


def vit_l_16(weight_path: str, num_classes: int, device: torch.device = torch.device("cpu")):
    """
    Load and return a pre-trained ViT-L/16 model.

    Parameters:
        weight_path (str): Path to the pre-trained weight file.

    Returns:
        torch.nn.Module: Pre-trained ViT-L/16 model with loaded weights.
    """
    # setup ViT model instance with pretrained weight
    pretrained_vit = timm.create_model(
        "vit_large_patch16_224", pretrained=False, num_classes=num_classes)

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
