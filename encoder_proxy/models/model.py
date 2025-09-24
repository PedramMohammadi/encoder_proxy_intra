import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# GDN Layer (Generalized Divisive Normalization)
# ------------------------------
class GDN(nn.Module):
    def __init__(self, channels, inverse=False):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_unconstrained = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma_unconstrained = nn.Parameter(torch.eye(channels).view(channels, channels, 1, 1))

    def forward(self, x):
        beta  = F.softplus(self.beta_unconstrained) + 1e-6
        gamma = F.softplus(self.gamma_unconstrained) + 1e-6
        norm = F.conv2d(x**2, gamma, bias=None) + beta
        norm = torch.clamp(norm, min=1e-12)
        norm = torch.sqrt(norm)
        return (x * norm) if self.inverse else (x / norm)

# ------------------------------
# FiLM Layer for CRF Conditioning
# ------------------------------
class FiLM(nn.Module):
    def __init__(self, in_channels, hidden_dim=32):
        super(FiLM, self).__init__()
        self.gamma = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )
        self.beta = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )

    def forward(self, x, crf):
        # x: (B, C, H, W); crf: (B, 1)
        gamma = self.gamma(crf).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta(crf).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

# ------------------------------
# Residual Encoder Block
# ------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ResidualConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gdn = GDN(out_channels)
        self.dropout = nn.Dropout(p=0.1)
        
        # Residual connection; use 1x1 conv if stride or channels change
        if stride != 1 or in_channels != out_channels:
            self.res = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
        else:
            self.res = nn.Identity()

    def forward(self, x):
        residual = self.res(x)
        out = self.conv(x)
        out = self.gdn(out)
        out = self.dropout(out)
        return out + residual  # Add residual

# ------------------------------
# Residual Decoder Block
# ------------------------------
class ResidualDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(ResidualDeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.gdn = GDN(out_channels, inverse=True)
        self.dropout = nn.Dropout(p=0.1)

        # Residual connection; use 1x1 convtranspose if stride or channels change
        if stride != 1 or in_channels != out_channels:
            self.res = nn.ConvTranspose2d(in_channels, out_channels, 1, stride, 0, output_padding)
        else:
            self.res = nn.Identity()

    def forward(self, x):
        residual = self.res(x)
        out = self.deconv(x)
        out = self.gdn(out)
        out = self.dropout(out)
        return out + residual  # Add residual

# ------------------------------
# Encoder Proxy Model
# ------------------------------
class EncoderProxy(nn.Module):

    def __init__(self, bottleneck_channels, film_hidden_dim):  # Config dict for hyperparameters
        super(EncoderProxy, self).__init__()

        # Default encoder/decoder channel configs
        enc_ch = [1, 64, 128, 256, bottleneck_channels]
        dec_ch = [bottleneck_channels, 256, 128, 64, 1]

        self.film_hidden_dim = film_hidden_dim

        # For later use in encoder/decoder
        self.enc_ch = enc_ch
        self.dec_ch = dec_ch

        # Build encoder with configurable channels
        encoder_blocks = []
        for i in range(len(enc_ch) - 1):
            encoder_blocks.append(ResidualConvBlock(enc_ch[i], enc_ch[i+1]))
        self.encoder = nn.Sequential(*encoder_blocks)

        self.film = FiLM(in_channels=enc_ch[-1], hidden_dim=self.film_hidden_dim)

        # Build decoder with configurable channels
        decoder_blocks = []
        for i in range(len(dec_ch) - 2):  # Up to last deconv
            decoder_blocks.append(ResidualDeconvBlock(dec_ch[i], dec_ch[i+1]))
        # Final deconv without block (as it's to 1 channel, no GDN)
        decoder_blocks.append(nn.ConvTranspose2d(dec_ch[-2], dec_ch[-1], kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_blocks.append(nn.Sigmoid())  # Output in [0,1]
        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, x, crf):
        # x: (B, 1, H, W); crf: (B, 1)
        enc = self.encoder(x)
        enc_film = self.film(enc, crf)

        q = torch.round(enc_film)
        enc_film = enc_film + (q - enc_film).detach()

        y_hat = self.decoder(enc_film)
        return y_hat