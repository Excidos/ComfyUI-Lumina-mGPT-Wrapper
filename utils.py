import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
import comfy.model_management as mm
import folder_paths
import logging
import json
from PIL import Image
import numpy as np
import yaml
from typing import List, Union
import math
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

import traceback

def download_and_load_model(model_name, precision, target_size):
    model_path = os.path.join(folder_paths.models_dir, "lumina_mgpt", model_name.split("/")[-1])
    
    if not os.path.exists(model_path):
        logger.info(f"Downloading Lumina-mGPT model to: {model_path}")
        snapshot_download(repo_id=model_name,
                          local_dir=model_path,
                          ignore_patterns=['*.md', '*.txt'],
                          local_dir_use_symlinks=False)

    required_files = [
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            logger.error(f"Required file {file} not found in {model_path}")
            raise FileNotFoundError(f"Required file {file} not found")

    tokenizer_files = ["text_tokenizer.json", "vqgan.yaml", "vqgan.ckpt"]
    tokenizer_path = os.path.join(model_path, 'ckpts', 'chameleon', 'tokenizer')
    
    os.makedirs(tokenizer_path, exist_ok=True)
    
    for file in tokenizer_files:
        file_path = os.path.join(tokenizer_path, file)
        if not os.path.exists(file_path):
            logger.info(f"Downloading {file}...")
            try:
                hf_hub_download(repo_id=model_name, filename=f"tokenizer/{file}", local_dir=tokenizer_path)
            except Exception as e:
                logger.error(f"Failed to download {file}: {str(e)}")
                raise

    try:
        inference_solver = CustomFlexARInferenceSolver(
            model_path=model_path,
            precision=precision,
            target_size=target_size,
            tokenizer_path=tokenizer_path
        )
        logger.info(f"Successfully created CustomFlexARInferenceSolver")
        return inference_solver
    except Exception as e:
        logger.error(f"Error creating CustomFlexARInferenceSolver: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * \
               torch.mean((z_q - z.detach())**2)
        
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)
        
        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)
        z_q = z_q.view(shape)
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q


class VQModel(nn.Module):
    def __init__(self, ckpt_path=None, n_embed=8192, embed_dim=256, **kwargs):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, 3, stride=2, padding=1)
        ])
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(embed_dim, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1)
        ])
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        
        new_state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("encoder."):
                new_k = k.replace("encoder.", "encoder.")
                new_state_dict[new_k] = v
            elif k.startswith("decoder."):
                new_k = k.replace("decoder.", "decoder.")
                new_state_dict[new_k] = v
            elif k.startswith("quantize."):
                new_state_dict[k] = v
        
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded VQModel from: {path}")

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        quant, emb_loss, info = self.quantize(x)
        return quant, emb_loss, info

    def decode(self, quant):
        for layer in self.decoder:
            quant = layer(quant)
        return quant

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)

        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class ChameleonConfig:
    def __init__(self, **kwargs):
        self.pad_token_id = kwargs.get('pad_token_id', 0)  # Default to 0 if not provided
        self.eos_token_id = kwargs.get('eos_token_id', 2)  # Default to 2 if not provided
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, model_path):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class ImageTokenizer:
    def __init__(self, cfg_path: str, ckpt_path: str, device: str | torch.device | None = None):
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        params = config["model"]["params"]
        if "lossconfig" in params:
            del params["lossconfig"]
        
        params["ckpt_path"] = ckpt_path
        params["n_embed"] = 8192  # Add this line
        params["embed_dim"] = 256  # Add this line
        
        self._vq_model = VQModel(**params)
        self._vq_model.eval()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._vq_model.to(self._device)
        
        self._dtype = next(self._vq_model.parameters()).dtype

    def _whiten_transparency(self, img: Image.Image) -> Image.Image:
        if img.mode == "RGB":
            return img
        vals_rgba = np.array(img.convert("RGBA"))
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")
        alpha = vals_rgba[:, :, 3] / 255.0
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
        return Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def img_tokens_from_pil(self, img: Image.Image) -> torch.Tensor:
        img = self._whiten_transparency(img)
        np_img = np.array(img) / 255.0
        np_img = np_img * 2 - 1
        img = torch.from_numpy(np_img).permute(2, 0, 1).to(self._device, dtype=self._dtype)
        img = img.unsqueeze(0)
        with torch.no_grad():
            _, _, [_, _, img_toks] = self._vq_model.encode(img)
        return img_toks

    def pil_from_img_toks(self, tokens: torch.Tensor, h_latent_dim=32, w_latent_dim=32) -> Image.Image:
        tokens = tokens.to(self._device)
        emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
        with torch.no_grad():
            codebook_entry = self._vq_model.quantize.get_codebook_entry(tokens, (1, h_latent_dim, w_latent_dim, emb_dim))
            pixels = self._vq_model.decode(codebook_entry)
        return self._pil_from_chw_tensor(pixels[0])

    def _pil_from_chw_tensor(self, chw_tensor: torch.Tensor) -> Image.Image:
        detached_chw_tensor = chw_tensor.detach().cpu()
        normalized_chw_tensor = (torch.clamp(detached_chw_tensor, -1.0, 1.0) + 1.0) / 2.0
        hwc_array = normalized_chw_tensor.permute(1, 2, 0).numpy()
        image_array_uint8 = (hwc_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array_uint8)
        return pil_image.convert("RGB")

class CustomLogitsProcessor:
    def __init__(self, cfg, image_top_k, text_top_k, vocab_size):
        self.cfg = cfg
        self.image_top_k = image_top_k
        self.text_top_k = text_top_k
        self.vocab_size = vocab_size

    def __call__(self, input_ids, scores):
        scores = self.cfg * scores + (1 - self.cfg) * torch.randn_like(scores)
        top_k = self.image_top_k if self.is_generating_image(input_ids) else self.text_top_k
        top_k = min(top_k, scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores[indices_to_remove] = float('-inf')
        return scores

    def is_generating_image(self, input_ids):
        image_start_token_id = 50277  # This is the token ID for "<|startofimage|>"
        return input_ids[0][-1].item() == image_start_token_id

class CustomFlexARItemProcessor:
    def __init__(self, tokenizer_path, target_size=768):
        self.tokenizer_path = tokenizer_path
        self.target_size = target_size

        with open(os.path.join(self.tokenizer_path, "text_tokenizer.json"), 'r', encoding='utf-8') as f:
            vocab_map = json.load(f)["model"]["vocab"]
        
        self.vocab_info = VocabInfo(vocab_map)
        self.vocab_translation = VocabTranslation(self.vocab_info)

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(self.tokenizer_path, "text_tokenizer.json"))

        self.image_start_token = "<|startofimage|>"
        self.image_end_token = "<|endofimage|>"
        self.new_line_token = "<|newline|>"

        self.image_tokenizer = ImageTokenizer(
            cfg_path=os.path.join(self.tokenizer_path, "vqgan.yaml"),
            ckpt_path=os.path.join(self.tokenizer_path, "vqgan.ckpt")
        )

    def process_item(self, item):
        if isinstance(item, str):
            return self.tokenizer.encode(item.replace("\n", self.new_line_token))
        elif isinstance(item, Image.Image):
            img_tokens = self.image_tokenizer.img_tokens_from_pil(item)
            bpe_tokens = self.vocab_translation.convert_img2bp2(img_tokens)
            return self.tokenizer.encode(f"{self.image_start_token}") + bpe_tokens.tolist() + self.tokenizer.encode(f"{self.image_end_token}")
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")

    def decode_image(self, tokens):
        image_tokens = []
        in_image = False
        for token in tokens:
            if token == self.tokenizer.encode(self.image_start_token)[0]:
                in_image = True
            elif token == self.tokenizer.encode(self.image_end_token)[0]:
                in_image = False
            elif in_image:
                image_tokens.append(token)
        
        img_tokens = self.vocab_translation.convert_bpe2img(torch.tensor(image_tokens))
        return self.image_tokenizer.pil_from_img_toks(img_tokens, h_latent_dim=self.target_size//8, w_latent_dim=self.target_size//8)

class ChameleonXLLMXForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self._device = device
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed_tokens(input_ids)
        
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return logits, loss

    def generate(self, input_ids, max_length, do_sample, temperature, num_return_sequences, logits_processor):
        input_ids = input_ids.to(self.device)
        batch_size = input_ids.shape[0]
        
        input_ids = input_ids.repeat(num_return_sequences, 1)
        
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=self.device)
        
        while True:
            logits, _ = self(input_ids)
            next_token_logits = logits[:, -1, :]
            
            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)
            
            if do_sample:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            next_tokens = next_tokens * unfinished_sequences + (1 - unfinished_sequences) * self.config.pad_token_id
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            unfinished_sequences = unfinished_sequences.mul((next_tokens != self.config.eos_token_id).long())
            
            if unfinished_sequences.max() == 0 or input_ids.shape[-1] >= max_length:
                break
        
        return input_ids

class CustomFlexARInferenceSolver:
    def __init__(self, model_path, precision, target_size, tokenizer_path):
        self.model_path = model_path
        self.precision = precision
        self.target_size = target_size
        self.tokenizer_path = tokenizer_path
        
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = ChameleonConfig.from_pretrained(model_path)
        self.model = ChameleonXLLMXForConditionalGeneration(config)
        self.model.to(self.device)  # First, move the model to the correct device
        self.model = self.model.to(self.dtype)  # Then, convert to the correct dtype
        self.model.eval()
        
        self.item_processor = CustomFlexARItemProcessor(tokenizer_path, target_size)

class CustomFlexARInferenceSolver:
    def __init__(self, model_path, precision, target_size, tokenizer_path):
        self.model_path = model_path
        self.precision = precision
        self.target_size = target_size
        self.tokenizer_path = tokenizer_path
        
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = ChameleonConfig.from_pretrained(model_path)
        self.model = ChameleonXLLMXForConditionalGeneration(config)
        self.model.to(self.device)  # First, move the model to the correct device
        self.model = self.model.to(self.dtype)  # Then, convert to the correct dtype
        self.model.eval()
        
        self.item_processor = CustomFlexARItemProcessor(tokenizer_path, target_size)

    def generate(self, images, qas, max_gen_len, temperature, logits_processor=None, num_inference_steps=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        logger.info(f"Starting generation with max_gen_len={max_gen_len}, temperature={temperature}")
        
        processed_qas = [[self.item_processor.process_item(q), self.item_processor.process_item(a) if a else None] for q, a in qas]
        processed_images = [self.item_processor.process_item(img) for img in images]
        
        input_ids = torch.tensor(processed_qas[0][0], device=self.device).unsqueeze(0)
        
        generate_kwargs = {
            "max_length": max_gen_len,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "num_return_sequences": 1,
            "logits_processor": logits_processor,
        }
        
        try:
            logger.info("Starting model generation")
            outputs = self.model.generate(input_ids=input_ids, **generate_kwargs)
            logger.info("Model generation completed")
        except Exception as e:
            logger.error(f"Error during model generation: {str(e)}")
            logger.error(f"Device: {self.device}, Model device: {self.model.device}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        generated_text = self.item_processor.tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        
        generated_image = None
        image_start_token = self.item_processor.tokenizer.encode(self.item_processor.image_start_token)[0]
        image_end_token = self.item_processor.tokenizer.encode(self.item_processor.image_end_token)[0]
        
        image_tokens = []
        in_image = False
        for token in outputs[0]:
            if token == image_start_token:
                in_image = True
            elif token == image_end_token:
                in_image = False
                break
            elif in_image:
                image_tokens.append(token.item())
        
        if image_tokens:
            try:
                logger.info("Decoding generated image")
                image_tokens_tensor = torch.tensor(image_tokens, device=self.device)
                generated_image = self.item_processor.decode_image(image_tokens_tensor)
                logger.info("Image decoding completed")
            except Exception as e:
                logger.error(f"Error decoding image: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                generated_image = None
        
        return generated_text, [generated_image] if generated_image else []
        
class VocabInfo:
    def __init__(self, vocab_map: dict[str, int]):
        self.name2val = vocab_map
        self.bos_id = vocab_map.get("<s>")
        self.eos_id = vocab_map.get("</s>")
        self.boi_id = vocab_map.get("<racm3:break>")
        self.eoi_id = vocab_map.get("<eoss>")
        self.pad_id = vocab_map.get("<pad>")
        self.eot_id = vocab_map.get("<reserved08706>")

    @property
    def val2name(self) -> dict[int, str]:
        return {v: k for k, v in self.name2val.items()}

    @property
    def image_tokens(self) -> list[int]:
        return sorted([val for name, val in self.name2val.items() if name.startswith("IMGIMG")])

class VocabTranslation:
    def __init__(self, vocab_info: VocabInfo):
        self._vocab = vocab_info

    @property
    def bpe2img(self) -> dict[int, int]:
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

        def remap(old_name: str) -> str:
            return "".join(img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1])

        return {tok: int(remap(self._vocab.val2name[tok])) for tok in self._vocab.image_tokens}

    @property
    def img2bpe(self) -> dict[int, int]:
        return {v: k for k, v in self.bpe2img.items()}

    def convert_bpe2img(self, bpe_batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.bpe2img[token.item()] for token in bpe_batch if token.item() in self.bpe2img])

    def convert_img2bp2(self, img_batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.img2bpe[token.item()] for token in img_batch if token.item() in self.img2bpe])