"""
Módulo de modelos para el proyecto AI for Earthquake Response.

Este módulo implementa arquitecturas de deep learning para la detección
automática de daños en edificios tras terremotos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple, Union
import timm
from loguru import logger

from .config import ModelConfig


class DoubleConv(nn.Module):
    """Bloque de doble convolución para UNet."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Bloque de downsampling para UNet."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bloque de upsampling para UNet."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding para tamaños diferentes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Capa de salida para UNet."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Arquitectura UNet para segmentación de daños.
    
    Esta implementación está optimizada para imágenes satelitales multitemporales
    con 6 canales de entrada (3 pre + 3 post terremoto).
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        bilinear: bool = True,
        dropout: float = 0.1,
        attention: bool = False
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.dropout = dropout
        self.attention = attention

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        
        self.up1 = Up(features[3] * 2, features[3], bilinear)
        self.up2 = Up(features[3], features[2], bilinear)
        self.up3 = Up(features[2], features[1], bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        self.outc = OutConv(features[0], out_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout2d(dropout)
        
        # Attention mechanism (opcional)
        if attention:
            self.attention1 = nn.MultiheadAttention(features[3] * 2, num_heads=8, batch_first=True)
            self.attention2 = nn.MultiheadAttention(features[3], num_heads=8, batch_first=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Aplicar dropout
        x5 = self.dropout_layer(x5)
        
        # Aplicar attention si está habilitado
        if self.attention:
            # Reshape para attention
            b, c, h, w = x5.shape
            x5_flat = x5.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
            x5_attended, _ = self.attention1(x5_flat, x5_flat, x5_flat)
            x5 = x5_attended.transpose(1, 2).view(b, c, h, w)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Aplicar attention en el decoder si está habilitado
        if self.attention:
            b, c, h, w = x.shape
            x_flat = x.view(b, c, h * w).transpose(1, 2)
            x_attended, _ = self.attention2(x_flat, x_flat, x_flat)
            x = x_attended.transpose(1, 2).view(b, c, h, w)
        
        logits = self.outc(x)
        return logits


class ResNetUNet(nn.Module):
    """
    UNet con encoder basado en ResNet.
    
    Utiliza un backbone ResNet pre-entrenado como encoder y añade
    un decoder UNet para la segmentación.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        in_channels: int = 6,
        out_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super(ResNetUNet, self).__init__()
        
        # Cargar backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Backbone {backbone} no soportado")
        
        # Modificar primera capa para 6 canales
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Extraer capas del encoder
        self.encoder1 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu
        )
        self.encoder2 = self.backbone.layer1
        self.encoder3 = self.backbone.layer2
        self.encoder4 = self.backbone.layer3
        self.encoder5 = self.backbone.layer4
        
        # Decoder UNet
        self.decoder4 = Up(encoder_channels[4], encoder_channels[3])
        self.decoder3 = Up(encoder_channels[3], encoder_channels[2])
        self.decoder2 = Up(encoder_channels[2], encoder_channels[1])
        self.decoder1 = Up(encoder_channels[1], encoder_channels[0])
        
        # Capa de salida
        self.outc = OutConv(encoder_channels[0], out_channels)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Aplicar dropout
        e5 = self.dropout(e5)
        
        # Decoder
        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        
        # Salida
        logits = self.outc(d1)
        return logits


class EfficientNetUNet(nn.Module):
    """
    UNet con encoder basado en EfficientNet.
    
    Utiliza EfficientNet como backbone para mejor eficiencia computacional.
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_channels: int = 6,
        out_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super(EfficientNetUNet, self).__init__()
        
        # Cargar backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True
        )
        
        # Obtener canales de salida del backbone
        encoder_channels = self.backbone.feature_info.channels()
        
        # Decoder UNet
        self.decoder4 = Up(encoder_channels[4], encoder_channels[3])
        self.decoder3 = Up(encoder_channels[3], encoder_channels[2])
        self.decoder2 = Up(encoder_channels[2], encoder_channels[1])
        self.decoder1 = Up(encoder_channels[1], encoder_channels[0])
        
        # Capa de salida
        self.outc = OutConv(encoder_channels[0], out_channels)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # Encoder
        features = self.backbone(x)
        
        # Aplicar dropout al feature más profundo
        features[4] = self.dropout(features[4])
        
        # Decoder
        d4 = self.decoder4(features[4], features[3])
        d3 = self.decoder3(d4, features[2])
        d2 = self.decoder2(d3, features[1])
        d1 = self.decoder1(d2, features[0])
        
        # Salida
        logits = self.outc(d1)
        return logits


class SegmentationTransformer(nn.Module):
    """
    Transformer para segmentación de daños.
    
    Implementa un modelo basado en transformers para la segmentación
    de imágenes satelitales.
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super(SegmentationTransformer, self).__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 32, 32))  # Para 512x512
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Decoder para segmentación
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 16, out_channels, kernel_size=1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Decoder
        logits = self.decoder(x)
        return logits


class TransformerBlock(nn.Module):
    """Bloque transformer individual."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Reshape para attention
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
        
        # Self-attention
        x_flat = x_flat + self._attention_block(self.norm1(x_flat))
        
        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        return x
    
    def _attention_block(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output


def get_model(config: ModelConfig) -> nn.Module:
    """
    Crear modelo según la configuración.
    
    Args:
        config: Configuración del modelo
        
    Returns:
        Modelo de PyTorch
    """
    
    if config.model_name == "unet":
        model = UNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            features=[64, 128, 256, 512],
            bilinear=True,
            dropout=config.dropout,
            attention=config.attention
        )
    
    elif config.model_name == "resnet_unet":
        model = ResNetUNet(
            backbone=config.backbone,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            pretrained=config.pretrained,
            dropout=config.dropout
        )
    
    elif config.model_name == "efficientnet_unet":
        model = EfficientNetUNet(
            backbone=config.backbone,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            pretrained=config.pretrained,
            dropout=config.dropout
        )
    
    elif config.model_name == "transformer":
        model = SegmentationTransformer(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            dropout=config.dropout
        )
    
    else:
        raise ValueError(f"Modelo {config.model_name} no soportado")
    
    logger.info(f"Modelo creado: {config.model_name}")
    logger.info(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Contar parámetros del modelo.
    
    Args:
        model: Modelo de PyTorch
        
    Returns:
        Diccionario con estadísticas de parámetros
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    } 