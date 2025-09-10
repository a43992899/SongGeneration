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
from torch.utils.data import Dataset, DataLoader

from marble.core.utils import instantiate_from_config
from marble.core.base_encoder import BaseEncoder


def load_model(config_path, ckpt_path, device='cpu'):
    """
    加载模型。如果指定的 ckpt_path 不存在，则从 Hugging Face 下载。
    """
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型配置
    model = instantiate_from_config(config['model'])
    
    # 加载模型检查点
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"Unexpected keys in state_dict: {unexpected}")
    
    model.to(device)
    model.eval()
    return model


class VQEncoder(BaseEncoder):
    NAME = "VQEncoder"
    TOKEN_RATE = 25  # Number of feature frames per second of audio
    SAMPLING_RATE = 24000  # Audio sampling rate expected by the model
    NUM_FEATURES = 1024  # Hidden dimension of the VQ model
    
    def __init__(
        self, 
        config_path: str, 
        ckpt_path: str, 
        use_ema: bool = False,
        mode: str = "vq_emb",  # one of ["vq_emb", "indices", "pre_vq_emb"]
        offload_teachers: bool = True
    ) -> None:
        super().__init__()
        self.model = load_model(config_path, ckpt_path)
        self.sample_rate = self.SAMPLING_RATE
        self.use_ema = use_ema
        if self.use_ema:
            print("Using EMA model")
            self.model.model_ema.store(self.model.parameters())
            self.model.model_ema.copy_to(self.model)
        
        assert mode in ["vq_emb", "indices", "pre_vq_emb"], f"Invalid mode: {mode}, must be one of ['vq_emb', 'indices', 'pre_vq_emb']"
        self.mode = mode
        
        
        self.offload_teachers = offload_teachers
        if self.offload_teachers:
            self.model.teachers = None  
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入音频进行编码，返回量化后的特征和隐藏状态。
        
        Args:
            audio (torch.Tensor): 输入音频张量，形状为 (batch_size, channels, time).
        
        Returns:
            indices: [B, T]
            quant_s: tuple of [B, T, D] contains single element
            h_s: tuple of [B, T, D] contains single element
        """
        quant_s, h_s, emb_loss, indices = self.model.encode_student(audio, layer_zero_only=False)
        
        if self.mode == "indices":
            return indices
        
        if self.mode == "pre_vq_emb":
            return (h_s.squeeze(1),)
        
        return (quant_s.squeeze(1),)

if __name__ == "__main__":
    # Example usage
    config_path = 'yrb/exp3.4.1.yaml'
    ckpt_path = 'checkpoints/exp3.4.1/epoch=49-step=10850.ckpt'
    
    encoder = VQEncoder(config_path, ckpt_path, use_ema=False, mode='indices')
    
    # Create a dummy audio tensor (batch_size=1, channels=1, time=72000)
    audio = torch.randn(1, 1, 72000)
    
    # Forward pass
    indices = encoder(audio).squeeze()  # Shape: (n_q, T)
    print("Quantized shape:", indices.shape)
    