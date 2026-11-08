"""Harmonic mixture-of-experts with a learned one-step phase harmonizer.

The module contains only the architecture and differentiable optical system. Distributed
training, datasets, checkpointing, and Kaggle orchestration live in the training notebook.
"""

import math

import torch
from torch import nn
from pytorch_msssim import ssim

IMAGE_SIZE = 128
WAVELENGTH = 532e-9
PIXEL_PITCH = 10e-6
ZS = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
SSIM_WEIGHT = 1.5
GATE_BALANCE_WEIGHT = 0.01

class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())

class UpBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__(); self.reduce = nn.Conv2d(cin, cout, 1); self.block = DoubleConv(cout + skip, cout)
    def forward(self, x, skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.block(torch.cat((self.reduce(x), skip), dim=1))

class DepthEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__(); hidden = max(16, channels // 2)
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, channels))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, depths):
        return self.net(depths[:, None])[:, :, None, None]

class PlaneFusion(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(4 * channels, channels, 1, bias=False), nn.GroupNorm(8, channels), nn.SiLU())

class SpatialExpert(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        hidden = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, channels, 1, bias=False),
        )
    def forward(self, x):
        return x + self.net(x)

class CrossPlaneTransformerExpert(nn.Module):
    def __init__(self, channels=512, planes=4, height=8, width=8, dim=256):
        super().__init__()
        self.planes, self.height, self.width = planes, height, width
        self.input_projection = nn.Linear(channels, dim)
        self.position = nn.Parameter(torch.zeros(1, planes * height * width, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=3 * dim, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.output_projection = nn.Linear(dim, channels)
        self.fusion = PlaneFusion(channels)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, plane_features):
        batch, planes, channels, height, width = plane_features.shape
        if (planes, height, width) != (self.planes, self.height, self.width):
            raise RuntimeError(f'Unexpected bottleneck shape: {plane_features.shape}')
        tokens = plane_features.permute(0, 1, 3, 4, 2).reshape(batch, planes * height * width, channels)
        tokens = self.input_projection(tokens) + self.position
        tokens = self.output_projection(self.transformer(tokens))
        maps = tokens.reshape(batch, planes, height, width, channels).permute(0, 1, 4, 2, 3)
        return self.fusion(maps.reshape(batch, planes * channels, height, width))

class FourierExpert(nn.Module):
    def __init__(self, channels=512, planes=4, spectral_channels=128, height=8, width=8):
        super().__init__()
        self.planes = planes
        self.reduce = nn.Conv2d(channels, spectral_channels, 1, bias=False)
        self.weight_real = nn.Parameter(torch.ones(1, spectral_channels, height, width // 2 + 1))
        self.weight_imag = nn.Parameter(torch.zeros(1, spectral_channels, height, width // 2 + 1))
        self.expand = nn.Sequential(
            nn.Conv2d(spectral_channels, channels, 1, bias=False), nn.GroupNorm(8, channels), nn.SiLU(),
        )
        self.fusion = PlaneFusion(channels)

    def forward(self, plane_features):
        batch, planes, channels, height, width = plane_features.shape
        values = self.reduce(plane_features.reshape(batch * planes, channels, height, width))
        spectrum = torch.fft.rfft2(values.float(), norm='ortho')
        spectral_weight = torch.complex(self.weight_real, self.weight_imag)
        values = torch.fft.irfft2(spectrum * spectral_weight, s=(height, width), norm='ortho')
        values = self.expand(values.to(plane_features.dtype))
        values = values.reshape(batch, planes, channels, height, width)
        return self.fusion(values.reshape(batch, planes * channels, height, width))

class HarmonicMoEUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.planes = 4
        channels = (32, 64, 128, 256, 512)
        self.encoders = nn.ModuleList([
            DoubleConv(1, 32), DoubleConv(32, 64), DoubleConv(64, 128),
            DoubleConv(128, 256), DoubleConv(256, 512),
        ])
        self.depth_embeddings = nn.ModuleList([DepthEmbedding(c) for c in channels])
        self.fusions = nn.ModuleList([PlaneFusion(c) for c in channels])
        self.pool = nn.MaxPool2d(2)
        self.spatial_expert = SpatialExpert(512)
        self.transformer_expert = CrossPlaneTransformerExpert(512)
        self.fourier_expert = FourierExpert(512)
        self.gate = nn.Sequential(nn.Linear(16 + 3 * 512, 256), nn.SiLU(), nn.Linear(256, 3))
        self.d3, self.d2 = UpBlock(512, 256, 256), UpBlock(256, 128, 128)
        self.d1, self.d0 = UpBlock(128, 64, 64), UpBlock(64, 32, 32)
        self.head = nn.Conv2d(32, 1, 1)
        z = torch.tensor(ZS, dtype=torch.float32) * 1e3
        self.register_buffer('depths', (z - z.mean()) / (z.max() - z.min()))

    @staticmethod
    def image_statistics(x):
        image = (x + 1) * 0.5
        mean = image.mean(dim=(-2, -1))
        std = image.std(dim=(-2, -1))
        dark = (image < 0.05).to(image.dtype).mean(dim=(-2, -1))
        dx = (image[..., :, 1:] - image[..., :, :-1]).abs().mean(dim=(-2, -1))
        dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(dim=(-2, -1))
        edge = 0.5 * (dx + dy)
        return torch.stack((mean, std, dark, edge), dim=-1).flatten(1)

    def forward(self, x):
        batch, planes, height, width = x.shape
        features = x.reshape(batch * planes, 1, height, width)
        depths = self.depths.repeat(batch)
        skips = []
        plane_bottleneck = None
        for level, (encoder, embedding, fusion) in enumerate(zip(self.encoders, self.depth_embeddings, self.fusions)):
            features = encoder(features) + embedding(depths).to(features.dtype)
            _, channels, h, w = features.shape
            planes_view = features.reshape(batch, planes, channels, h, w)
            if level < 4:
                skips.append(fusion(planes_view.reshape(batch, planes * channels, h, w)))
                features = self.pool(features)
            else:
                plane_bottleneck = planes_view

        plain_fused = self.fusions[4](plane_bottleneck.reshape(batch, planes * 512, 8, 8))
        expert_outputs = torch.stack((
            self.spatial_expert(plain_fused),
            self.transformer_expert(plane_bottleneck),
            self.fourier_expert(plane_bottleneck),
        ), dim=1)
        pooled = expert_outputs.mean(dim=(-2, -1)).flatten(1)
        gates = torch.softmax(self.gate(torch.cat((self.image_statistics(x), pooled), dim=1)), dim=1)
        fused = (expert_outputs * gates[:, :, None, None, None]).sum(dim=1)
        e0, e1, e2, e3 = skips
        fused = self.d3(fused, e3); fused = self.d2(fused, e2)
        fused = self.d1(fused, e1); fused = self.d0(fused, e0)
        return self.head(fused), gates

class FresnelLayer(nn.Module):
    def __init__(self, device):
        super().__init__(); c = (torch.arange(IMAGE_SIZE, dtype=torch.float64) - IMAGE_SIZE / 2) * PIXEL_PITCH
        u, v = torch.meshgrid(c, c, indexing='ij')
        kernels = [torch.exp(1j * math.pi * (u.square() + v.square()) / (WAVELENGTH * z)).to(torch.complex64) for z in ZS]
        self.register_buffer('kernels', torch.stack(kernels).to(device), persistent=False)
    def forward(self, phase):
        field = torch.polar(torch.ones_like(phase, dtype=torch.float32), phase.float()); outputs = []
        for kernel in self.kernels:
            value = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field * kernel, dim=(-2, -1))), dim=(-2, -1)).abs()
            outputs.append(value)
        amplitude = torch.cat(outputs, dim=1)
        return amplitude / (amplitude.amax(dim=(-2, -1), keepdim=True) + 1e-8)

def pearson_cc(a, b, eps=1e-8):
    a = a - a.mean(dim=(-2, -1), keepdim=True); b = b - b.mean(dim=(-2, -1), keepdim=True)
    return (a * b).sum(dim=(-2, -1)) / (torch.sqrt(a.square().sum(dim=(-2, -1)) * b.square().sum(dim=(-2, -1))) + eps)

def doe_metrics(target, reconstruction):
    cc = pearson_cc(target, reconstruction)
    sv = torch.stack([
        ssim(target[:, m:m+1], reconstruction[:, m:m+1], data_range=1.0, size_average=False)
        for m in range(4)
    ], dim=1)
    return cc, sv

class PhaseHarmonizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0, self.e1, self.e2 = DoubleConv(13, 32), DoubleConv(32, 64), DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 128)
        self.u2, self.u1, self.u0 = UpBlock(128, 128, 128), UpBlock(128, 64, 64), UpBlock(64, 32, 32)
        self.head = nn.Conv2d(32, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad(): self.head.bias[0] = 1.0

    def forward(self, target, initial_reconstruction, initial_phase):
        error = target - initial_reconstruction
        x = torch.cat((target, initial_reconstruction, error, initial_phase / math.pi), dim=1)
        e0 = self.e0(x); e1 = self.e1(self.pool(e0)); e2 = self.e2(self.pool(e1))
        x = self.bottleneck(self.pool(e2)); x = self.u2(x, e2); x = self.u1(x, e1); x = self.u0(x, e0)
        raw = self.head(x); norm = torch.sqrt(raw[:, :1].square() + raw[:, 1:].square() + 1e-8)
        return raw[:, :1] / norm, raw[:, 1:] / norm

class EndToEndHarmonized(nn.Module):
    def __init__(self, device):
        super().__init__(); self.backbone = HarmonicMoEUNet(); self.harmonizer = PhaseHarmonizer()
        self.physics = FresnelLayer(device)

    def synthesize(self, target):
        target = target.float()
        with torch.autocast(device_type='cuda'):
            logits, gates = self.backbone(2 * target - 1)
            initial_phase = math.pi * torch.tanh(logits)
        with torch.autocast(device_type='cuda', enabled=False):
            initial_reconstruction = self.physics(initial_phase.float())
        with torch.autocast(device_type='cuda'):
            correction_real, correction_imag = self.harmonizer(target, initial_reconstruction, initial_phase)
        base = torch.polar(torch.ones_like(initial_phase, dtype=torch.float32), initial_phase.float())
        correction = torch.complex(correction_real.float(), correction_imag.float())
        final_field = base * correction
        final_phase = torch.atan2(final_field.imag, final_field.real)
        with torch.autocast(device_type='cuda', enabled=False):
            final_reconstruction = self.physics(final_phase)
        return initial_phase, initial_reconstruction, final_phase, final_reconstruction, gates, correction_real

    def forward(self, target):
        initial_phase, initial_reconstruction, final_phase, final_reconstruction, gates, correction_real = self.synthesize(target)
        target = target.float(); initial_cc, initial_ssim = doe_metrics(target, initial_reconstruction)
        final_cc, final_ssim = doe_metrics(target, final_reconstruction)
        initial_loss = (1-initial_cc).mean() + SSIM_WEIGHT * (1-initial_ssim).mean()
        final_loss = (1-final_cc).mean() + SSIM_WEIGHT * (1-final_ssim).mean()
        correction_penalty = (1-correction_real).mean()
        mean_gate = gates.float().mean(0); balance = (mean_gate-1/3).square().sum()
        total = final_loss + 0.2*initial_loss + 0.01*correction_penalty + GATE_BALANCE_WEIGHT*balance
        return total, final_cc, final_ssim, gates, correction_penalty, initial_cc, initial_ssim

__all__ = [
    "SpatialExpert",
    "CrossPlaneTransformerExpert",
    "FourierExpert",
    "HarmonicMoEUNet",
    "PhaseHarmonizer",
    "FresnelLayer",
    "EndToEndHarmonized",
]
