import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol
from PIL import Image
import importlib
from dataclasses import dataclass
from typing import Optional
from omegaconf import OmegaConf
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

def parse_configs(config_path: str):
    """Load a config file and return component sections as DictConfigs."""
    config = OmegaConf.load(config_path)
    rae_config = config.get("stage_1", None)
    return rae_config

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = RAE(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # see if it's a ckpt from training by checking for "model"
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            raise NotImplementedError("Loading from 'model' key not implemented yet.")
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        print(f'target {config["target"]} loaded from {ckpt_path}')
    return model

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class RAE(nn.Module):
    def __init__(self, 
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)
        # see if the encoder has patch size attribute            
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # normalize input
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
    
if __name__ == "__main__":
    
    satellite_image = Image.open("/users/x/z/xzhang31/scratch/VIGOR/Chicago/satellite/satellite_41.88190866564266_-87.6312772533304.png").convert('RGB')

    ground_image = Image.open("/users/x/z/xzhang31/scratch/VIGOR/Chicago/panorama/-2gorwEtLdVp_uCrO1yHfQ,41.881952,-87.631132,.jpg")

    transformed_satellite = transform(satellite_image).unsqueeze(0).to("cuda")
    transformed_ground = transform(ground_image).unsqueeze(0).to("cuda")
    
    
    # simple test
    config = parse_configs("./RAE/configs/DINOv2-B.yaml")
    # print(config)
    model = instantiate_from_config(config).to("cuda").eval()
    with torch.no_grad():
        x_latent = model.encode(transformed_satellite)
        x_rec = model.decode(x_latent)
        
    print(x_latent.shape)
    # transformed_satellite = transformed_satellite / 2 + 0.5
    # transformed_ground = transformed_ground / 2 + 0.5
    torchvision.utils.save_image(transformed_satellite, "original.png")
    torchvision.utils.save_image(x_rec, "reconstructed.png")
    
    print(x_rec.shape)