"""
Model factory: pretrained backbone from timm, fine-tuned for binary
real-vs-fake classification.
"""

import timm
import torch.nn as nn


def build_model(backbone: str = "resnet50", pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    backbone: any timm model name, e.g. "resnet50", "efficientnet_b0", "vit_small_patch16_224"
    freeze_backbone: if True, only the classification head is trainable
                      (useful for the first few warmup epochs)
    """
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=2)

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Keep the final classifier head trainable; freeze everything else.
            if "fc" not in name and "classifier" not in name and "head" not in name:
                param.requires_grad = False

    return model


def set_backbone_trainable(model: nn.Module, trainable: bool = True) -> None:
    """Unfreeze all layers -- call this after your warmup epochs to fine-tune the full network."""
    for param in model.parameters():
        param.requires_grad = trainable
