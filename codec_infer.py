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
from third_party.demucs.models.pretrained import get_model_from_yaml
from third_party.demucs.models.apply import apply_model


class Separator:
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            # self.device = torch.device(f"cuda:{gpu_id}")
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def load_audio(self, f):
        a, fs = torchaudio.load(f)
        if (fs != 48000):
            a = torchaudio.functional.resample(a, fs, 48000)
        return a

    def run(self, waveform, original_sr=48000):
        waveform_ = torchaudio.functional.resample(waveform, original_sr, self.demucs_model.samplerate)

        ref = waveform_.mean(0)
        waveform_ -= ref.mean()
        waveform_ /= ref.std()
        sources = apply_model(self.demucs_model, waveform_[None], device=self.device, shifts=1, split=True, overlap=0.25,
                              progress=True, num_workers=0, segment=None)[0]
        sources *= ref.std()
        sources += ref.mean()
        vocal_audio = sources[self.demucs_model.sources.index('vocal')]
        # rescale
        vocal_audio = vocal_audio / max(1.01 * vocal_audio.abs().max(), 1)
        vocal_audio = torchaudio.functional.resample(vocal_audio, self.demucs_model.samplerate, original_sr)

        assert abs(vocal_audio.shape[-1] - waveform.shape[-1]) <= 1
        waveform = waveform[..., :min(vocal_audio.shape[-1], waveform.shape[-1])]
        vocal_audio = vocal_audio[..., :min(vocal_audio.shape[-1], waveform.shape[-1])]
        bgm_audio = waveform - vocal_audio
        return vocal_audio, bgm_audio


class MucodecTokenizer:
    def __init__(self, cfg,
                 use_mix_tokenizer=True,
                 use_dual_tokenizer=True
                ) -> None:
        self.cfg = cfg
        self.use_mix_tokenizer = use_mix_tokenizer
        self.use_dual_tokenizer = use_dual_tokenizer

        if use_mix_tokenizer:
            self.audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
            for param in self.audio_tokenizer.parameters():
                param.requires_grad = False
        else:
            self.audio_tokenizer = None
        if use_dual_tokenizer:
            assert "audio_tokenizer_checkpoint_sep" in self.cfg.keys()
            self.separate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            for param in self.separate_tokenizer.parameters():
                param.requires_grad = False
            self.separator = Separator()
        else:
            self.separate_tokenizer = None
    
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
            tokens, scale = self.audio_tokenizer.encode(src_wav)
            encode_tokens.append(tokens)
        if self.use_dual_tokenizer:
            if 'vocal_audio' in kwargs:
                vocal_wav = self.read_audio(kwargs['vocal_audio'])
                src_wav = src_wav[..., :min(vocal_wav.shape[-1], src_wav.shape[-1])]
                vocal_wav = vocal_wav[..., :min(vocal_wav.shape[-1], src_wav.shape[-1])]
                bgm_wav = src_wav - vocal_wav
                del src_wav
                torch.cuda.empty_cache()
            else:
                vocal_wav, bgm_wav = self.separator.run(src_wav)
                del src_wav
                torch.cuda.empty_cache()
            # torchaudio.save(f'outputs5/vocal_{os.path.basename(src_audio)}', vocal_wav.cpu().float(), 48000)
            # torchaudio.save(f'outputs5/bgm_{os.path.basename(src_audio)}', bgm_wav.cpu().float(), 48000)
            vocal_tokens, bgm_tokens = self.separate_tokenizer.encode(vocal_wav, bgm_wav)
            del vocal_wav, bgm_wav
            torch.cuda.empty_cache()
            assert len(vocal_tokens.shape) == len(bgm_tokens.shape) == 3, \
                f"vocal and bgm tokens should have a shape [B, C, T]! " \
                f"got vocal len={vocal_tokens.shape}, and bgm len={bgm_tokens.shape}"
            assert vocal_tokens.shape[-1] == bgm_tokens.shape[-1], \
                f"vocal and bgm tokens should have the same length! " \
                f"got vocal len={vocal_tokens.shape[-1]}, and bgm len={bgm_tokens.shape[-1]}, src_audio={bgm_wav.shape} {src_wav.shape} {vocal_wav.shape}"
            encode_tokens.append(torch.cat([vocal_tokens, bgm_tokens], dim=1))
        token_seq_len = np.min([e.shape[-1] for e in encode_tokens])
        encode_tokens = [e[..., :token_seq_len] for e in encode_tokens]
        return torch.cat(encode_tokens, dim=1)

    def decoder(self, gen_tokens, is_separate=True):
        if is_separate:
            assert gen_tokens.shape[1] == 2
            vocal_prompt = bgm_prompt = None
            gen_tokens_vocal = gen_tokens[:, [0], :]
            gen_tokens_bgm = gen_tokens[:, [1], :]
            gen_audio_seperate = self.separate_tokenizer.decode([gen_tokens_vocal, gen_tokens_bgm], vocal_prompt, bgm_prompt)
            return gen_audio_seperate
        else:
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
        use_mix_tokenizer=True,
        use_dual_tokenizer=True
    )

    tokens = model.encoder(src_audio)
    assert tokens.shape[1] == 3

    os.makedirs(f'{save_dir}', exist_ok=True)
    gen_audio = model.decoder(tokens[:, :1], is_separate=False)
    torchaudio.save(f'{save_dir}/{os.path.basename(src_audio).split(".")[0]}_recon_wo_sep.wav', gen_audio[0].cpu().float(), cfg.sample_rate)
    gen_audio = model.decoder(tokens[:, 1:], is_separate=True)
    torchaudio.save(f'{save_dir}/{os.path.basename(src_audio).split(".")[0]}_recon_w_sep.wav', gen_audio[0].cpu().float(), cfg.sample_rate)


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
        use_dual_tokenizer=True
    )
    print("tokens.shape:", tokens.shape)
    # assert tokens.shape[1] == 3
    if tokens.shape[1] == 3:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gen_audio = model.decoder(tokens[:, :1], is_separate=False)
        torchaudio.save(f'{save_path}_recon_wo_sep.wav', gen_audio[0].cpu().float(), cfg.sample_rate)
        gen_audio = model.decoder(tokens[:, 1:], is_separate=True)
        torchaudio.save(f'{save_path}_recon_w_sep.wav', gen_audio[0].cpu().float(), cfg.sample_rate)
    elif tokens.shape[1] == 2:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # gen_audio = model.decoder(tokens[:, :1], is_separate=False)
        # torchaudio.save(f'{save_path}_recon_wo_sep.wav', gen_audio[0].cpu().float(), cfg.sample_rate)
        gen_audio = model.decoder(tokens[:, :], is_separate=True)
        torchaudio.save(f'{save_path}_recon_w_sep.wav', gen_audio[0].cpu().float(), cfg.sample_rate)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
    np.random.seed(int(time.time()))
    ckpt_path = "./ckpt/songgeneration_base"
    save_dir = "./tmp"
    src_audio = "sample/sample_prompt_audio.wav"
    # ckpt_path = sys.argv[1]
    # src_audio = sys.argv[2]
    # save_dir = sys.argv[3]
    
    codec_infer(
        ckpt_path=ckpt_path,
        save_dir=save_dir,
        src_audio=src_audio,
    )

    # idx = '000005FX0vVDAw'

    # for i in range(3, 10):
    #     for j in range(2):
    #         dual_npy_filename = f"tmp/idx-long_00{str(i)}-dual-{str(j)}.temp80.cfgl1.3.First50_180M.dual.token.npy"
    #         dual_npy_tokens = np.load(dual_npy_filename)
    #         if dual_npy_tokens.shape[0] % 2 != 0:
    #             dual_npy_tokens = dual_npy_tokens[:-1]
    #         dual_tracks_tokens = rearrange(dual_npy_tokens, "(n d) -> d n", d=2) # 0: vocal, 1: instr
    #         print("dual_tracks_tokens shape:", dual_tracks_tokens.shape)
    #         if max(dual_tracks_tokens[0]) > 16383:
    #             print("vocal range not valid!", max(dual_tracks_tokens[0]))
    #         if max(dual_tracks_tokens[1]) > 16383:
    #             print("instr range not valid!", max(dual_tracks_tokens[1]))
    #             dual_tracks_tokens[1] = dual_tracks_tokens[1] - 16384
    #             print("instr range after fix:", max(dual_tracks_tokens[1]))
    #         np.save(dual_npy_filename.replace("dual.token.npy", "vocal.token.npy"), dual_tracks_tokens[0])
    #         np.save(dual_npy_filename.replace("dual.token.npy", "intru.token.npy"), dual_tracks_tokens[1])
    # # vocal npy and instrument npy
    # # v_npy_filename = "/opt/tiger/ziyaz/byteYuE/finetune/inference/test/0815-finetune/stage1/pop,-国语_tp0@93_T1_rp1@1_maxtk3000_3fb678c5-d247-4ac2-a032-dc3e5b3a90f1_vtrack.npy"
    # # i_npy_filename = "/opt/tiger/ziyaz/byteYuE/finetune/inference/test/0815-finetune/stage1/pop,-国语_tp0@93_T1_rp1@1_maxtk3000_3fb678c5-d247-4ac2-a032-dc3e5b3a90f1_itrack.npy"
    # # tokens = np.stack(
    # #     [
    # #      np.load(v_npy_filename),
    # #      np.load(v_npy_filename),
    # #      np.load(i_npy_filename),
    # #     ],
    # #     axis=1
    # # )
    #         tokens = np.stack(
    #             [
    #                 dual_tracks_tokens[0].reshape(1, dual_tracks_tokens[0].shape[0]),
    #                 dual_tracks_tokens[0].reshape(1, dual_tracks_tokens[1].shape[0]),
    #                 dual_tracks_tokens[1].reshape(1, dual_tracks_tokens[1].shape[0])
    #             ],
    #             axis=1
    #         )

    #         print(tokens.shape, tokens.dtype)
    #         print(tokens)
    #         # tokens[tokens == 16383] = 16382
    #         # breakpoint()
    #         decoder_audios(
    #             tokens=torch.Tensor(tokens).long(),
    #             ckpt_path=ckpt_path,
    #             save_path=f'{save_dir}/{dual_npy_filename.split("/")[-1].replace(".token.npy", "_180M")}.wav'
    #        )
