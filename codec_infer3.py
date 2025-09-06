import sys
import os

import time
import json
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
from codeclm.models import builders


class MucodecTokenizer:
    def __init__(self, cfg,
                 use_mix_tokenizer=True
                ) -> None:
        self.cfg = cfg
        self.use_mix_tokenizer = use_mix_tokenizer

        if use_mix_tokenizer:
            self.audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
            for param in self.audio_tokenizer.parameters():
                param.requires_grad = False
        else:
            self.audio_tokenizer = None
    
    def read_audio(self, audio):
        if isinstance(audio, str):
            src_wav, sr = torchaudio.load(audio)
            if (sr != self.cfg.sample_rate):
                src_wav = torchaudio.functional.resample(src_wav, sr, self.cfg.sample_rate)
        else:
            if audio.ndim == 3:
                audio = audio.squeeze(0)
            src_wav = audio
        if src_wav.shape[0] == 1:
            # dual track
            src_wav = torch.cat([src_wav, src_wav], dim=0)
        return src_wav

    def encoder(self, src_audio, **kwargs):
        src_wav = self.read_audio(src_audio)

        encode_tokens = []
        if self.use_mix_tokenizer:
            tokens, scale = self.audio_tokenizer.encode_latent(src_wav)
            encode_tokens.append(tokens)
        token_seq_len = np.min([e.shape[-1] for e in encode_tokens])
        encode_tokens = [e[..., :token_seq_len] for e in encode_tokens]
        return torch.cat(encode_tokens, dim=1)

    def decoder(self, gen_tokens):
        assert gen_tokens.shape[1] == 1
        prompt = None
        gen_audio = self.audio_tokenizer.decode(gen_tokens, prompt)
        return gen_audio


def codec_infer(
        src_audio='sample/sample_prompt_audio.wav',
        ckpt_path='ckpt',
        save_dir='tmp',
    ):
    cfg_path = os.path.join(ckpt_path, 'config.yaml')
    ckpt_path = os.path.join(ckpt_path, 'model.pt')
    cfg = OmegaConf.load(cfg_path)
    cfg.mode = 'inference'

    model = MucodecTokenizer(
        cfg,
        use_mix_tokenizer=True
    )

    tokens = model.encoder(src_audio)



def decoder_audios(
        tokens,
        ckpt_path='ckpt',
        save_path='tmp',):
    cfg_path = os.path.join(ckpt_path, 'config.yaml')
    ckpt_path = os.path.join(ckpt_path, 'model.pt')
    cfg = OmegaConf.load(cfg_path)
    cfg.mode = 'inference'

    model = MucodecTokenizer(
        cfg,
        use_mix_tokenizer=True,
        use_dual_tokenizer=False
    )
    print("tokens.shape:", tokens.shape)
    if tokens.shape[1] == 3:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gen_audio = model.decoder(tokens[:, :1])
        torchaudio.save(f'{save_path}_recon_wo_sep.wav', gen_audio[0].cpu().float(), cfg.sample_rate)
    elif tokens.shape[1] == 2:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
    np.random.seed(int(time.time()))
    ckpt_path = "./ckpt/songgeneration_base"
    save_dir = "./tmp"
    src_audio = "sample/0027_blackened.wav"
    
    codec_infer(
        ckpt_path=ckpt_path,
        save_dir=save_dir,
        src_audio=src_audio,
    )