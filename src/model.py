"""
Módulo de modelos para el proyecto AI for Earthquake Response.

Este módulo implementa arquitecturas de deep learning para la detección
automática de daños en edificios tras terremotos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from loguru import logger

from config import ModelConfig


class MaskedSpatialAttention(nn.Module):
    """
    Atención espacial suave: de una máscara binaria 1×H×W genera un
    mapa de atención C×H×W que modula el feature‐map.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # Conv 1×1 de 1 canal (la máscara) → in_channels
        self.attn_conv = nn.Conv2d(1, in_channels, kernel_size=1)
        
        # Inicialización para que arranque en modo "casi uniforme"
        nn.init.kaiming_normal_(self.attn_conv.weight, mode='fan_in', nonlinearity='sigmoid')
        if self.attn_conv.bias is not None:
            nn.init.zeros_(self.attn_conv.bias)

    def forward(self, feat_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: [B, 1, H, W], feat_map: [B, C, H, W]
        # 1) redimensionar la máscara
        mask_resized = F.interpolate(mask, size=feat_map.shape[-2:],
                                     mode='bilinear', align_corners=False)
        # 2) Aplicar dropout espacial para robustez ante máscaras ruidosas
        mask_dropped = F.dropout2d(mask_resized, p=0.1, training=self.training)
        # 3) generar mapa de atención y suavizarlo
        attn = torch.sigmoid(self.attn_conv(mask_dropped))  # [B, C, H, W]
        # 4) aplicar gating espacial
        return feat_map * attn


class FPN(nn.Module):
    """
    Feature Pyramid Network para extraer características en múltiples escalas.
    """
    def __init__(self, in_channels_list, out_channels):
        """
        in_channels_list: lista [C2, C3, C4, C5] de tu backbone
        out_channels: número de canales que unifica cada P_i (p.ej. 256)
        """
        super().__init__()
        # 1×1 laterales
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        # 3×3 para refinar tras la suma top-down
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        """
        features: [C5, C4, C3, C2] (de mayor a menor resolución)
        returns: [P5, P4, P3, P2] (cada uno con out_channels)
        """
        # top layer
        last_inner = self.lateral_convs[0](features[0])  # C5 → P5_inner
        results = [self.smooth_convs[0](last_inner)]   # produce P5

        # recorre C4, C3, C2
        for idx, feature in enumerate(features[1:], 1):
            lateral = self.lateral_convs[idx](feature)
            # upsample del último inner
            inner = lateral + F.interpolate(last_inner, scale_factor=2, mode='nearest')
            results.append(self.smooth_convs[idx](inner))
            last_inner = inner

        return results  # [P5, P4, P3, P2]


class CNNBackbone(nn.Module):
    """
    Backbone CNN para extraer características de una imagen.
    """
    def __init__(self, in_channels, dropout=0.2, base=64):
        super().__init__()
        self.base = base
        
        # Definir stages individuales para FPN
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base*2),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base*2),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base*4),
            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base*4)
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base*4),
            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base*4)
        )
        
        # Para compatibilidad con el modelo original
        self.features = nn.Sequential(
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Canales de salida de cada stage para FPN
        self.stage1_out = base      # C2
        self.stage2_out = base * 2  # C3
        self.stage3_out = base * 4  # C4
        self.stage4_out = base * 4  # C5
        
        self.out_dim = base * 4
        # Inicialización
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B, out_dim)
        return x
    
    def forward_stages(self, x):
        """
        Extrae características de cada stage para FPN.
        Returns: [C2, C3, C4, C5] (de menor a mayor resolución)
        """
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return [c2, c3, c4, c5]


class SiameseCNNClassifier(nn.Module):
    """
    Modelo siamés mejorado con backbone CNN compartido para pre y post imagen.
    Entrada: [B, 2*C, H, W] (pre + post concatenados), y building_mask [B, 2, H, W]
    Salida: logit por tile (clasificación binaria)
    """
    def __init__(self, in_channels, dropout=0.3, base=64, backbone_weights_path=None):
        super().__init__()
        self.c = in_channels  # canales por imagen (pre o post)
        # El backbone ahora espera solo los canales de imagen (sin máscara concatenada)
        self.backbone = CNNBackbone(in_channels, dropout=dropout, base=base)
        if backbone_weights_path is not None:
            state_dict = torch.load(backbone_weights_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Pesos del backbone cargados desde {backbone_weights_path}")
        
        # --- PARTIR EL BACKBONE EN "ENCODER" + "POOL" Y CREAR LA ATENCIÓN ---
        # 1) Extraer los módulos de features (todo menos el AdaptiveAvgPool2d final)
        modules = list(self.backbone.features)
        if not isinstance(modules[-1], nn.AdaptiveAvgPool2d):
            raise RuntimeError("Se esperaba AdaptiveAvgPool2d al final de backbone.features")
        self.backbone_encoder = nn.Sequential(*modules[:-1])
        self.backbone_pool = modules[-1]  # AdaptiveAvgPool2d(1)

        # 2) Instanciar el módulo de atención en base a la dimensión de salida del encoder
        attn_channels = self.backbone.out_dim  # base*4 en tu CNNBackbone
        self.attn = MaskedSpatialAttention(attn_channels)
        
        feat_dim = self.backbone.out_dim
        # MLP más profundo y con BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 3, 256),  # 3: pre, post, abs(pre-post)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Logit
        )

    def forward(self, x, building_mask=None):
        pre = x[:, :self.c, :, :]
        post = x[:, self.c:, :, :]

        # Si no hay máscara, creamos una de unos
        if building_mask is None:
            mask_pre = torch.ones_like(pre[:, :1, :, :])
            mask_post = torch.ones_like(post[:, :1, :, :])
        else:
            # Normalizar la máscara a shape [B, 2, H, W] y extraer siempre [0] y [1]
            if building_mask.shape[1] != 2:
                raise ValueError(f"building_mask debe tener 2 canales, pero tiene {building_mask.shape[1]}")
            mask_pre = building_mask[:, 0:1, :, :]
            mask_post = building_mask[:, 1:2, :, :]

        # 1) Extraer feature‐map antes del pooling
        fmap_pre = self.backbone_encoder(pre)   # [B, C, H', W']
        fmap_post = self.backbone_encoder(post)  # [B, C, H', W']

        # 2) Aplicar atención espacial
        fmap_pre = self.attn(fmap_pre, mask_pre)
        fmap_post = self.attn(fmap_post, mask_post)

        # 3) Pooling y vectorizar
        pooled_pre = self.backbone_pool(fmap_pre)   # [B, C, 1, 1]
        pooled_post = self.backbone_pool(fmap_post)  # [B, C, 1, 1]
        feat_pre = pooled_pre.view(pre.size(0), -1)  # [B, C]
        feat_post = pooled_post.view(post.size(0), -1) # [B, C]

        # 4) Resto igual: diff, concat y classifier
        feat_diff = torch.abs(feat_pre - feat_post)
        feat = torch.cat([feat_pre, feat_post, feat_diff], dim=1)
        logit = self.classifier(feat)
        return logit.squeeze(1)


class SiameseCNNClassifierFPN(nn.Module):
    """
    Modelo siamés con FPN para capturar características en múltiples escalas.
    Entrada: [B, 2*C, H, W] (pre + post concatenados), y building_mask [B, 2, H, W]
    Salida: logit por tile (clasificación binaria)
    """
    def __init__(self, in_channels, dropout=0.3, base=64, backbone_weights_path=None, out_channels_fpn=256):
        super().__init__()
        self.c = in_channels  # canales por imagen (pre o post)
        
        # 1) Cargar backbone
        self.backbone = CNNBackbone(in_channels, dropout=dropout, base=base)
        if backbone_weights_path is not None:
            state_dict = torch.load(backbone_weights_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Pesos del backbone cargados desde {backbone_weights_path}")
        
        # 2) Saca los canales de cada stage
        fpn_in_channels = [
            self.backbone.stage1_out,  # C2
            self.backbone.stage2_out,  # C3
            self.backbone.stage3_out,  # C4
            self.backbone.stage4_out,  # C5
        ]
        
        # 3) Instancia el FPN
        self.fpn = FPN(fpn_in_channels, out_channels_fpn)
        
        # 4) Ajusta el clasificador: ahora recibirá 3×(4 niveles)×out_channels_fpn
        fpn_feat_dim = 4 * out_channels_fpn
        # tras concatenar pre, post y diff:
        self.classifier = nn.Sequential(
            nn.Linear(3 * fpn_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Logit
        )

    def _split_masks(self, building_mask):
        """Extrae máscaras pre/post de building_mask [B, 2, H, W]"""
        if building_mask is None:
            # Crear máscaras de unos con el tamaño correcto
            # Necesitamos inferir el tamaño de la imagen
            return None, None
        else:
            if building_mask.shape[1] != 2:
                raise ValueError(f"building_mask debe tener 2 canales, pero tiene {building_mask.shape[1]}")
            mask_pre = building_mask[:, 0:1, :, :]
            mask_post = building_mask[:, 1:2, :, :]
            return mask_pre, mask_post

    def _pool_levels(self, p_list, mask=None):
        """
        Hace pooling sobre cada nivel de la pirámide.
        p_list: [P5, P4, P3, P2] (de menor a mayor resolución)
        mask: [B, 1, H, W] o None
        """
        out_feats = []
        for p in p_list:
            # Si hay máscara, aplicarla antes del pooling
            if mask is not None:
                mask_rs = F.interpolate(mask, size=p.shape[-2:], mode='bilinear', align_corners=False)
                p = p * mask_rs
            pooled = F.adaptive_avg_pool2d(p, 1).view(p.size(0), -1)
            out_feats.append(pooled)
        return torch.cat(out_feats, dim=1)  # [B, 4*out_channels_fpn]

    def forward(self, x, building_mask=None):
        pre = x[:, :self.c, :, :]
        post = x[:, self.c:, :, :]

        # 1) Saca C2…C5 de cada rama
        feats_pre = self.backbone.forward_stages(pre)   # [C2p, C3p, C4p, C5p]
        feats_post = self.backbone.forward_stages(post)  # [C2t, C3t, C4t, C5t]

        # 2) Aplica el FPN: => [P5, P4, P3, P2]
        p_pre = self.fpn(feats_pre[::-1])   # invertimos a [C5,C4,C3,C2]
        p_post = self.fpn(feats_post[::-1])

        # 3) Extraer máscaras pre/post
        mask_pre, mask_post = self._split_masks(building_mask)

        # 4) Pooling sobre cada nivel con máscaras opcionales
        feat_pre = self._pool_levels(p_pre, mask_pre)
        feat_post = self._pool_levels(p_post, mask_post)

        # 5) Construir diff y concatenar
        feat_diff = torch.abs(feat_pre - feat_post)
        feat = torch.cat([feat_pre, feat_post, feat_diff], dim=1)

        # 6) Clasificar
        logit = self.classifier(feat)
        return logit.squeeze(1)


class CorrelationSiameseHead(nn.Module):
    """
    Cabeza siamés basada en correlación espacial entre feature-maps.
    Calcula correlación en ventana fija entre F_pre y F_post.
    """
    def __init__(self, in_channels, correlation_window=5, out_channels=256, dropout=0.3):
        super().__init__()
        self.correlation_window = correlation_window
        self.corr_channels = correlation_window ** 2
        
        # Proyección de features
        self.proj_pre = nn.Conv2d(in_channels, out_channels, 1)
        self.proj_post = nn.Conv2d(in_channels, out_channels, 1)
        
        # Procesamiento de correlación
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(self.corr_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def correlation(self, f1, f2):
        """Calcula correlación espacial entre dos feature maps."""
        b, c, h, w = f1.shape
        pad = self.correlation_window // 2
        
        # Padding
        f1_pad = F.pad(f1, (pad, pad, pad, pad))
        f2_pad = F.pad(f2, (pad, pad, pad, pad))
        
        # Extraer patches
        patches1 = F.unfold(f1_pad, self.correlation_window, stride=1)
        patches2 = F.unfold(f2_pad, self.correlation_window, stride=1)
        
        # Reshape para correlación
        patches1 = patches1.view(b, c, self.corr_channels, h, w)
        patches2 = patches2.view(b, c, self.corr_channels, h, w)
        
        # Correlación
        corr = (patches1 * patches2).sum(dim=1)  # [B, corr_channels, H, W]
        return corr

    def forward(self, feat_pre, feat_post, fmap_pre, fmap_post):
        # Proyección
        f_pre = self.proj_pre(fmap_pre)
        f_post = self.proj_post(fmap_post)
        
        # Correlación
        corr = self.correlation(f_pre, f_post)
        
        # Procesamiento
        corr_feat = self.corr_encoder(corr)
        corr_feat = corr_feat.view(corr_feat.size(0), -1)
        
        # Clasificación
        logit = self.classifier(corr_feat)
        return logit.squeeze(1)


class CrossAttentionTransformerHead(nn.Module):
    """
    Cabeza basada en cross-attention transformer.
    Usa capas de cross-attention entre tokens pre y post.
    """
    def __init__(self, in_channels, embed_dim=256, num_heads=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Proyección a tokens
        self.proj_pre = nn.Conv2d(in_channels, embed_dim, 1)
        self.proj_post = nn.Conv2d(in_channels, embed_dim, 1)
        
        # Tokens CLS
        self.cls_token_pre = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_token_post = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, feat_pre, feat_post, fmap_pre, fmap_post):
        b = fmap_pre.size(0)
        
        # Proyección y reshape a tokens
        tokens_pre = self.proj_pre(fmap_pre).view(b, self.embed_dim, -1).transpose(1, 2)
        tokens_post = self.proj_post(fmap_post).view(b, self.embed_dim, -1).transpose(1, 2)
        
        # Añadir tokens CLS
        cls_pre = self.cls_token_pre.expand(b, -1, -1)
        cls_post = self.cls_token_post.expand(b, -1, -1)
        
        tokens_pre = torch.cat([cls_pre, tokens_pre], dim=1)
        tokens_post = torch.cat([cls_post, tokens_post], dim=1)
        
        # Concatenar para cross-attention
        all_tokens = torch.cat([tokens_pre, tokens_post], dim=1)
        
        # Aplicar transformer layers
        for layer in self.transformer_layers:
            all_tokens = layer(all_tokens)
        
        # Extraer CLS fusionado (promedio de ambos CLS)
        cls_fused = (all_tokens[:, 0] + all_tokens[:, tokens_pre.size(1)]) / 2
        
        # Clasificación
        logit = self.classifier(cls_fused)
        return logit.squeeze(1)


class MultiTaskHead(nn.Module):
    """
    Cabeza multi-task: clasificación + segmentación.
    Comparte features pero aprende dos tareas conjuntas.
    """
    def __init__(self, in_channels, dropout=0.3):
        super().__init__()
        
        # Rama de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 3, 256),  # pre, post, diff
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Rama de segmentación
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels * 2, 128, 3, padding=1),  # concat pre+post
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)  # salida binaria
        )

    def forward(self, feat_pre, feat_post, fmap_pre, fmap_post):
        # Clasificación
        feat_diff = torch.abs(feat_pre - feat_post)
        feat_concat = torch.cat([feat_pre, feat_post, feat_diff], dim=1)
        logit_cls = self.classifier(feat_concat)
        
        # Segmentación
        fmap_concat = torch.cat([fmap_pre, fmap_post], dim=1)
        logit_seg = self.segmentation_head(fmap_concat)
        
        return logit_cls.squeeze(1), logit_seg


class ASPPStyleHead(nn.Module):
    """
    Cabeza estilo ASPP (Atrous Spatial Pyramid Pooling).
    Aplica convoluciones dilatadas a múltiples escalas.
    """
    def __init__(self, in_channels, out_channels=256, dropout=0.3):
        super().__init__()
        
        # Proyección inicial
        self.proj = nn.Conv2d(in_channels * 3, out_channels, 1)  # pre, post, diff
        
        # Ramas ASPP
        self.aspp_branches = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 1),  # 1x1
            nn.Conv2d(out_channels, out_channels, 3, padding=6, dilation=6),   # rate=6
            nn.Conv2d(out_channels, out_channels, 3, padding=12, dilation=12), # rate=12
            nn.Conv2d(out_channels, out_channels, 3, padding=18, dilation=18), # rate=18
        ])
        
        # Pooling global
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fusión
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, feat_pre, feat_post, fmap_pre, fmap_post):
        # Concatenar feature maps
        fmap_diff = torch.abs(fmap_pre - fmap_post)
        fmap_concat = torch.cat([fmap_pre, fmap_post, fmap_diff], dim=1)
        
        # Proyección
        x = self.proj(fmap_concat)
        
        # Ramas ASPP
        aspp_out = []
        for branch in self.aspp_branches:
            aspp_out.append(branch(x))
        
        # Pooling global en una rama
        global_feat = self.global_pool(x)
        aspp_out.append(global_feat.expand_as(aspp_out[0]))
        
        # Fusión
        fused = torch.cat(aspp_out, dim=1)
        fused = self.fusion(fused)
        
        # Pooling final
        pooled = self.global_pool(fused).view(fused.size(0), -1)
        
        # Clasificación
        logit = self.classifier(pooled)
        return logit.squeeze(1)


class RelationNetworkHead(nn.Module):
    """
    Cabeza basada en Relation Network.
    Aprende relaciones no lineales entre vectores pre y post.
    """
    def __init__(self, in_channels, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Dimensión de entrada: [f_pre, f_post, f_pre*f_post, |f_pre-f_post|]
        relation_input_dim = in_channels * 4
        
        # MLP de relaciones
        self.relation_mlp = nn.Sequential(
            nn.Linear(relation_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, feat_pre, feat_post, fmap_pre, fmap_post):
        # Construir vector de relaciones
        feat_diff = torch.abs(feat_pre - feat_post)
        feat_prod = feat_pre * feat_post
        
        # Concatenar todas las relaciones
        relation_input = torch.cat([
            feat_pre, feat_post, feat_prod, feat_diff
        ], dim=1)
        
        # Procesar con MLP
        logit = self.relation_mlp(relation_input)
        return logit.squeeze(1)


class TemporalConv3DHead(nn.Module):
    """
    Cabeza basada en convoluciones 3D temporales.
    Trata pre/post como frames de un video.
    """
    def __init__(self, in_channels, out_channels=128, dropout=0.3):
        super().__init__()
        
        # Convoluciones 3D
        self.conv3d_1 = nn.Conv3d(in_channels, out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(out_channels)
        
        self.conv3d_2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(out_channels)
        
        # Pooling 3D
        self.pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, feat_pre, feat_post, fmap_pre, fmap_post):
        # Apilar como video: [B, C, T=2, H, W]
        video = torch.stack([fmap_pre, fmap_post], dim=2)
        
        # Convoluciones 3D
        x = F.relu(self.bn3d_1(self.conv3d_1(video)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        
        # Pooling 3D
        x = self.pool3d(x).view(x.size(0), -1)
        
        # Clasificación
        logit = self.classifier(x)
        return logit.squeeze(1)


class PrototypeBasedHead(nn.Module):
    """
    Cabeza basada en prototipos para few-shot learning.
    Calcula distancia a prototipos de daño/no-daño.
    """
    def __init__(self, in_channels, num_prototypes=10, dropout=0.3):
        super().__init__()
        self.num_prototypes = num_prototypes
        
        # Proyección de features
        self.proj = nn.Sequential(
            nn.Linear(in_channels * 3, 256),  # pre, post, diff
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        # Prototipos aprendibles
        self.prototypes_damaged = nn.Parameter(torch.randn(num_prototypes, 128))
        self.prototypes_intact = nn.Parameter(torch.randn(num_prototypes, 128))
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(2, 64),  # 2 distancias
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, feat_pre, feat_post, fmap_pre, fmap_post):
        # Proyección
        feat_diff = torch.abs(feat_pre - feat_post)
        feat_concat = torch.cat([feat_pre, feat_post, feat_diff], dim=1)
        projected = self.proj(feat_concat)
        
        # Calcular distancias a prototipos
        dist_damaged = torch.cdist(projected, self.prototypes_damaged).min(dim=1)[0]
        dist_intact = torch.cdist(projected, self.prototypes_intact).min(dim=1)[0]
        
        # Concatenar distancias
        distances = torch.stack([dist_intact, dist_damaged], dim=1)
        
        # Clasificación
        logit = self.classifier(distances)
        return logit.squeeze(1)


class EfficientNetB0Backbone(nn.Module):
    """
    Backbone EfficientNet-B0 compatible con la interfaz de CNNBackbone.
    Permite usar pesos preentrenados o inicialización desde cero.
    """
    def __init__(self, in_channels=3, dropout=0.2, pretrained=True):
        super().__init__()
        
        # Importar timm aquí para evitar dependencias innecesarias
        try:
            import timm
        except ImportError:
            raise ImportError("timm no está instalado. Instala con: pip install timm")
        
        # Crear modelo EfficientNet-B0
        self.efficientnet = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            drop_rate=dropout
        )
        
        # Obtener información de los stages
        feature_info = self.efficientnet.feature_info
        self.stage1_out = feature_info.channels()[0]  # C2
        self.stage2_out = feature_info.channels()[1]  # C3  
        self.stage3_out = feature_info.channels()[2]  # C4
        self.stage4_out = feature_info.channels()[3]  # C5
        
        # Dimensión de salida (usamos el último stage)
        self.out_dim = self.stage4_out
        
        # Para compatibilidad con CNNBackbone
        self.features = nn.Sequential(
            self.efficientnet,
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        """Forward completo con pooling para compatibilidad."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B, out_dim)
        return x
    
    def forward_stages(self, x):
        """
        Extrae características de cada stage para FPN.
        Returns: [C2, C3, C4, C5] (de menor a mayor resolución)
        """
        features = self.efficientnet(x)
        return features  # EfficientNet ya devuelve [C2, C3, C4, C5]


class SiameseEfficientNetClassifier(nn.Module):
    """
    Modelo siamés con backbone EfficientNet-B0.
    Entrada: [B, 2*C, H, W] (pre + post concatenados), y building_mask [B, 2, H, W]
    Salida: logit por tile (clasificación binaria)
    """
    def __init__(self, in_channels, dropout=0.3, pretrained=True, backbone_weights_path=None):
        super().__init__()
        self.c = in_channels  # canales por imagen (pre o post)
        
        # Backbone EfficientNet-B0
        self.backbone = EfficientNetB0Backbone(
            in_channels=in_channels, 
            dropout=dropout, 
            pretrained=pretrained
        )
        
        if backbone_weights_path is not None:
            state_dict = torch.load(backbone_weights_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Pesos del backbone cargados desde {backbone_weights_path}")
        
        # --- PARTIR EL BACKBONE EN "ENCODER" + "POOL" Y CREAR LA ATENCIÓN ---
        # Para EfficientNet, el encoder es el propio efficientnet
        self.backbone_encoder = self.backbone.efficientnet
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)

        # Instanciar el módulo de atención
        attn_channels = self.backbone.out_dim
        self.attn = MaskedSpatialAttention(attn_channels)
        
        feat_dim = self.backbone.out_dim
        # MLP más profundo y con BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 3, 256),  # 3: pre, post, abs(pre-post)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Logit
        )

    def forward(self, x, building_mask=None):
        pre = x[:, :self.c, :, :]
        post = x[:, self.c:, :, :]

        # Si no hay máscara, creamos una de unos
        if building_mask is None:
            mask_pre = torch.ones_like(pre[:, :1, :, :])
            mask_post = torch.ones_like(post[:, :1, :, :])
        else:
            # Normalizar la máscara a shape [B, 2, H, W] y extraer siempre [0] y [1]
            if building_mask.shape[1] != 2:
                raise ValueError(f"building_mask debe tener 2 canales, pero tiene {building_mask.shape[1]}")
            mask_pre = building_mask[:, 0:1, :, :]
            mask_post = building_mask[:, 1:2, :, :]

        # 1) Extraer feature‐map antes del pooling
        fmap_pre = self.backbone_encoder(pre)   # EfficientNet devuelve lista de features
        fmap_post = self.backbone_encoder(post)  # Tomamos el último (más profundo)
        
        # EfficientNet devuelve una lista, tomamos el último elemento
        if isinstance(fmap_pre, (list, tuple)):
            fmap_pre = fmap_pre[-1]   # [B, C, H', W']
            fmap_post = fmap_post[-1]  # [B, C, H', W']

        # 2) Aplicar atención espacial
        fmap_pre = self.attn(fmap_pre, mask_pre)
        fmap_post = self.attn(fmap_post, mask_post)

        # 3) Pooling y vectorizar
        pooled_pre = self.backbone_pool(fmap_pre)   # [B, C, 1, 1]
        pooled_post = self.backbone_pool(fmap_post)  # [B, C, 1, 1]
        feat_pre = pooled_pre.view(pre.size(0), -1)  # [B, C]
        feat_post = pooled_post.view(post.size(0), -1) # [B, C]

        # 4) Resto igual: diff, concat y classifier
        feat_diff = torch.abs(feat_pre - feat_post)
        feat = torch.cat([feat_pre, feat_post, feat_diff], dim=1)
        logit = self.classifier(feat)
        return logit.squeeze(1)


class SiameseEfficientNetFPN(nn.Module):
    """
    Modelo siamés con EfficientNet-B0 + FPN.
    Entrada: [B, 2*C, H, W] (pre + post concatenados), y building_mask [B, 2, H, W]
    Salida: logit por tile (clasificación binaria)
    """
    def __init__(self, in_channels, dropout=0.3, pretrained=True, backbone_weights_path=None, out_channels_fpn=256):
        super().__init__()
        self.c = in_channels  # canales por imagen (pre o post)
        
        # Backbone EfficientNet-B0
        self.backbone = EfficientNetB0Backbone(
            in_channels=in_channels, 
            dropout=dropout, 
            pretrained=pretrained
        )
        
        if backbone_weights_path is not None:
            state_dict = torch.load(backbone_weights_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Pesos del backbone cargados desde {backbone_weights_path}")
        
        # Saca los canales de cada stage
        fpn_in_channels = [
            self.backbone.stage1_out,  # C2
            self.backbone.stage2_out,  # C3
            self.backbone.stage3_out,  # C4
            self.backbone.stage4_out,  # C5
        ]
        
        # Instancia el FPN
        self.fpn = FPN(fpn_in_channels, out_channels_fpn)
        
        # Ajusta el clasificador: ahora recibirá 3×(4 niveles)×out_channels_fpn
        fpn_feat_dim = 4 * out_channels_fpn
        self.classifier = nn.Sequential(
            nn.Linear(3 * fpn_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Logit
        )

    def _split_masks(self, building_mask):
        """Extrae máscaras pre/post de building_mask [B, 2, H, W]"""
        if building_mask is None:
            return None, None
        else:
            if building_mask.shape[1] != 2:
                raise ValueError(f"building_mask debe tener 2 canales, pero tiene {building_mask.shape[1]}")
            mask_pre = building_mask[:, 0:1, :, :]
            mask_post = building_mask[:, 1:2, :, :]
            return mask_pre, mask_post

    def _pool_levels(self, p_list, mask=None):
        """
        Hace pooling sobre cada nivel de la pirámide.
        p_list: [P5, P4, P3, P2] (de menor a mayor resolución)
        mask: [B, 1, H, W] o None
        """
        out_feats = []
        for p in p_list:
            # Si hay máscara, aplicarla antes del pooling
            if mask is not None:
                mask_rs = F.interpolate(mask, size=p.shape[-2:], mode='bilinear', align_corners=False)
                p = p * mask_rs
            pooled = F.adaptive_avg_pool2d(p, 1).view(p.size(0), -1)
            out_feats.append(pooled)
        return torch.cat(out_feats, dim=1)  # [B, 4*out_channels_fpn]

    def forward(self, x, building_mask=None):
        pre = x[:, :self.c, :, :]
        post = x[:, self.c:, :, :]

        # 1) Saca C2…C5 de cada rama
        feats_pre = self.backbone.forward_stages(pre)   # [C2p, C3p, C4p, C5p]
        feats_post = self.backbone.forward_stages(post)  # [C2t, C3t, C4t, C5t]

        # 2) Aplica el FPN: => [P5, P4, P3, P2]
        p_pre = self.fpn(feats_pre[::-1])   # invertimos a [C5,C4,C3,C2]
        p_post = self.fpn(feats_post[::-1])

        # 3) Extraer máscaras pre/post
        mask_pre, mask_post = self._split_masks(building_mask)

        # 4) Pooling sobre cada nivel con máscaras opcionales
        feat_pre = self._pool_levels(p_pre, mask_pre)
        feat_post = self._pool_levels(p_post, mask_post)

        # 5) Construir diff y concatenar
        feat_diff = torch.abs(feat_pre - feat_post)
        feat = torch.cat([feat_pre, feat_post, feat_diff], dim=1)

        # 6) Clasificar
        logit = self.classifier(feat)
        return logit.squeeze(1)


def get_model(config: ModelConfig, backbone_weights_path=None) -> nn.Module:
    """
    Crear modelo según la configuración.
    Args:
        config: Configuración del modelo
        backbone_weights_path: ruta a pesos preentrenados para el backbone (opcional)
    Returns:
        Modelo de PyTorch
    """
    if config.model_name == "siamese_cnn_classifier":
        # Usar max_channels para el backbone (canales por imagen pre o post)
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        model = SiameseCNNClassifier(
            in_channels=in_channels,
            dropout=getattr(config, 'dropout', 0.2),
            backbone_weights_path=backbone_weights_path
        )
    elif config.model_name == "siamese_cnn_classifier_fpn":
        # Usar max_channels para el backbone (canales por imagen pre o post)
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        out_channels_fpn = getattr(config, 'out_channels_fpn', 256)
        model = SiameseCNNClassifierFPN(
            in_channels=in_channels,
            dropout=getattr(config, 'dropout', 0.2),
            backbone_weights_path=backbone_weights_path,
            out_channels_fpn=out_channels_fpn
        )
    elif config.model_name == "correlation_siamese_head":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        correlation_window = getattr(config, 'correlation_window', 5)
        out_channels = getattr(config, 'out_channels', 256)
        model = CorrelationSiameseHead(
            in_channels=in_channels,
            correlation_window=correlation_window,
            out_channels=out_channels,
            dropout=getattr(config, 'dropout', 0.2)
        )
    elif config.model_name == "cross_attention_transformer_head":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        embed_dim = getattr(config, 'embed_dim', 256)
        num_heads = getattr(config, 'num_heads', 8)
        num_layers = getattr(config, 'num_layers', 2)
        model = CrossAttentionTransformerHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=getattr(config, 'dropout', 0.2)
        )
    elif config.model_name == "multitask_head":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        model = MultiTaskHead(
            in_channels=in_channels,
            dropout=getattr(config, 'dropout', 0.2)
        )
    elif config.model_name == "aspp_style_head":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        out_channels = getattr(config, 'out_channels', 256)
        model = ASPPStyleHead(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=getattr(config, 'dropout', 0.2)
        )
    elif config.model_name == "relation_network_head":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        hidden_dim = getattr(config, 'hidden_dim', 256)
        model = RelationNetworkHead(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            dropout=getattr(config, 'dropout', 0.2)
        )
    elif config.model_name == "temporal_conv3d_head":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        out_channels = getattr(config, 'out_channels', 128)
        model = TemporalConv3DHead(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=getattr(config, 'dropout', 0.2)
        )
    elif config.model_name == "prototype_based_head":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        num_prototypes = getattr(config, 'num_prototypes', 10)
        model = PrototypeBasedHead(
            in_channels=in_channels,
            num_prototypes=num_prototypes,
            dropout=getattr(config, 'dropout', 0.2)
        )
    elif config.model_name == "siamese_efficientnet_classifier":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        pretrained = getattr(config, 'pretrained', True)
        model = SiameseEfficientNetClassifier(
            in_channels=in_channels,
            dropout=getattr(config, 'dropout', 0.2),
            pretrained=pretrained,
            backbone_weights_path=backbone_weights_path
        )
    elif config.model_name == "siamese_efficientnet_fpn":
        in_channels = getattr(config, 'max_channels', 3)
        backbone_weights_path = getattr(config, 'backbone_weights_path', None)
        pretrained = getattr(config, 'pretrained', True)
        out_channels_fpn = getattr(config, 'out_channels_fpn', 256)
        model = SiameseEfficientNetFPN(
            in_channels=in_channels,
            dropout=getattr(config, 'dropout', 0.2),
            pretrained=pretrained,
            backbone_weights_path=backbone_weights_path,
            out_channels_fpn=out_channels_fpn
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


if __name__ == '__main__':
    print("=" * 50)
    print("MODELOS DISPONIBLES EN model.py")
    print("=" * 50)
    print()
    
    print("MODELOS COMPLETOS:")
    models_complete = [
        "siamese_cnn_classifier",
        "siamese_cnn_classifier_fpn",
        "siamese_efficientnet_classifier",
        "siamese_efficientnet_fpn"
    ]
    
    for i, model_name in enumerate(models_complete, 1):
        print(f"{i}. {model_name}")
    
    print()
    print("CABEZAS ESPECIALIZADAS (solo cabecera):")
    models_heads = [
        "correlation_siamese_head",
        "cross_attention_transformer_head", 
        "multitask_head",
        "aspp_style_head",
        "relation_network_head",
        "temporal_conv3d_head",
        "prototype_based_head"
    ]
    
    for i, model_name in enumerate(models_heads, 1):
        print(f"{i}. {model_name}")
    
    print()
    print("=" * 50)
    print("Para usar un modelo, configura:")
    print("config.model_name = 'nombre_del_modelo'")
    print()
    print("NOTA: Las cabezas especializadas requieren un backbone")
    print("separado y se usan solo para la cabecera de clasificación.")
    print("=" * 50) 