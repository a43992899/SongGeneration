from typing import Sequence, Dict, Optional, Union, Tuple, List
import os
import json
import yaml
from tqdm import tqdm
from collections import defaultdict

import torch
import torchaudio
import torchaudio.transforms as T 
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from marble.core.utils import instantiate_from_config
from marble.core.base_encoder import BaseEncoder
from codeclm.models import builders


class MucodecTokenizer:
    def __init__(self, cfg,
                ) -> None:
        self.cfg = cfg
        self.audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
        for param in self.audio_tokenizer.parameters():
            param.requires_grad = False

    def encoder(self, src_audio, mode="pre_vq"):
        if type(src_audio) is str:
            src_audio = self.read_audio(src_audio)
        emb, _ = self.audio_tokenizer.encode_latent(src_audio, mode=mode)
        return emb

    def decoder(self, gen_tokens):
        assert gen_tokens.shape[1] == 1
        prompt = None
        gen_audio = self.audio_tokenizer.decode(gen_tokens, prompt)
        return gen_audio


def load_model(config_path, device='cpu'):
    """
    加载模型。如果指定的 ckpt_path 不存在，则从 Hugging Face 下载。
    """
    # 加载配置文件
    cfg = OmegaConf.load(config_path)
    cfg.mode = 'inference'
    
    # 加载模型配置
    model = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
    
    model.to(device)
    model.eval()
    return model


class VQEncoder(BaseEncoder):
    NAME = "VQEncoder"
    TOKEN_RATE = 25  # Number of feature frames per second of audio
    SAMPLING_RATE = 48000  # Audio sampling rate expected by the model
    NUM_FEATURES = 1024  # Hidden dimension of the VQ model
    
    def __init__(
        self, 
        config_path: str, 
        mode: str = "vq_emb",  # one of ["vq_emb", "indices", "pre_vq_emb"]
    ) -> None:
        super().__init__()
        self.model = load_model(config_path)
        self.sample_rate = self.SAMPLING_RATE
        
        assert mode in ["vq_emb", "pre_vq"], f"Invalid mode: {mode}, must be one of ['vq_emb', 'indices', 'pre_vq_emb']"
        self.mode = mode
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入音频进行编码，返回量化后的特征和隐藏状态。
        
        Args:
            audio (torch.Tensor): 输入音频张量，形状为 (batch_size, channels, time).
        
        Returns:
            tuple of [B, T, D] contains single element
        """
        emb, _ = self.model.encode_latent(audio, mode=self.mode)
        
        return emb

if __name__ == "__main__":
    # Example usage
    config_path = 'ckpt/songgeneration_base/config.yaml'
    
    encoder = VQEncoder(config_path, mode='vq_emb')
    
    # Create a dummy audio tensor (batch_size=1, channels=1, time=72000)
    audio = torch.randn(1, 1, 48000*5)
    
    # Forward pass
    emb = encoder(audio)[0] # Shape: (n_q, T)
    print("Quantized shape:", emb.shape)
    