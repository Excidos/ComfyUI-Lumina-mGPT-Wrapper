import os
import logging
import torch
import numpy as np
from PIL import Image
from lumina_mgpt.inference_solver import FlexARInferenceSolver
from lumina_mgpt.data.item_processor import FlexARItemProcessor, generate_crop_size_list

import folder_paths

logger = logging.getLogger(__name__)

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "Lumina-mGPT", "lumina_mgpt", "ckpts", "chameleon", "tokenizer")

class LuminamGPTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (["Alpha-VLLM/Lumina-mGPT-7B-512", "Alpha-VLLM/Lumina-mGPT-7B-768", "Alpha-VLLM/Lumina-mGPT-7B-768-Omni", "Alpha-VLLM/Lumina-mGPT-7B-1024"], {"default": "Alpha-VLLM/Lumina-mGPT-7B-768"}),
            "precision": (["bf16", "fp32"], {"default": "bf16"}),
        }}

    RETURN_TYPES = ("LUMINAMGPT", "INT")
    RETURN_NAMES = ("model", "target_size")
    FUNCTION = "load_model"
    CATEGORY = "LuminaWrapper"

    def load_model(self, model, precision):
        try:
            model_name = model.split("/")[-1]
            target_size = int(model_name.split("-")[-1])
            model_dir = os.path.join(folder_paths.models_dir, "lumina_mgpt", model_name)

            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

            required_files = [
                "config.json",
                "generation_config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json"
            ]

            for file in required_files:
                file_path = os.path.join(model_dir, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file not found: {file_path}")

            inference_solver = FlexARInferenceSolver(
                model_path=model_dir,
                precision=precision,
                target_size=target_size,
                tokenizer_path=TOKENIZER_PATH
            )
            logger.info(f"Successfully loaded model from {model_dir}")
            return (inference_solver, target_size)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

class LuminamGPTCropSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_size": (["512", "768", "1024"], {"default": "512"}),
                "aspectRatio": ([
                    "1:1 - square",
                    "4:3 - standard",
                    "16:9 - widescreen",
                    "2:3 - portrait",
                    "3:2 - landscape"
                ],)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("resolution",)
    FUNCTION = "select_crop"
    CATEGORY = "LuminaWrapper"

    def select_crop(self, target_size, aspectRatio):
        target_size = int(target_size)
        if aspectRatio == "1:1 - square":
            width = height = target_size
        elif aspectRatio == "4:3 - standard":
            width, height = int(target_size * 4 / 3), target_size
        elif aspectRatio == "16:9 - widescreen":
            width, height = int(target_size * 16 / 9), target_size
        elif aspectRatio == "2:3 - portrait":
            width, height = int(target_size * 2 / 3), target_size
        elif aspectRatio == "3:2 - landscape":
            width, height = target_size, int(target_size * 2 / 3)
        
        resolution = f"{width}x{height}"
        return (resolution,)

class LuminamGPTImageGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lumina_mgpt_model": ("LUMINAMGPT",),
                "prompt": ("STRING", {"multiline": True}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 16.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "image_top_k": ("INT", {"default": 4000, "min": 0, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "resolution": ("STRING", {"default": "512x512"}),
                "resolution_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "LuminaWrapper"

    def generate(self, lumina_mgpt_model, prompt, cfg, seed, image_top_k, temperature, resolution="512x512", resolution_input=None):
        try:
            # Use resolution_input if provided, otherwise use resolution
            resolution_to_use = resolution_input if resolution_input else resolution
            width, height = map(int, resolution_to_use.split('x'))

            if seed == 0:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)

            logits_processor = lumina_mgpt_model.create_logits_processor(cfg=cfg, text_top_k=5, image_top_k=image_top_k)

            full_prompt = f"Generate an image of {resolution_to_use} according to the following prompt:\n{prompt}"
            logger.info(f"Generating with prompt: {full_prompt}")

            generated_text, generated_images = lumina_mgpt_model.generate(
                images=[],
                qas=[[full_prompt, None]],
                max_gen_len=5000,
                temperature=temperature,
                logits_processor=logits_processor,
            )

            logger.info(f"Generation result type: {type(generated_images)}, content: {generated_images}")

            if not generated_images:
                logger.warning("No image was generated. Returning a blank image.")
                img_np = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                generated_image = generated_images[0]
                logger.info(f"Raw generated image type: {type(generated_image)}")
                logger.info(f"Raw generated image size: {generated_image.size}, mode: {generated_image.mode}")

                if generated_image.mode != 'RGB':
                    logger.warning(f"Image mode is {generated_image.mode}, converting to RGB.")
                    generated_image = generated_image.convert('RGB')

                generated_image = generated_image.resize((width, height), Image.LANCZOS)
                logger.info(f"Resized image size: {generated_image.size}, mode: {generated_image.mode}")

                img_np = np.array(generated_image)
                logger.info(f"NumPy array shape: {img_np.shape}, dtype: {img_np.dtype}")

            # Convert to tensor format
            image_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
            
            logger.info(f"Final image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            logger.info(f"Image tensor min: {image_tensor.min()}, max: {image_tensor.max()}")
            logger.info(f"Image tensor mean: {image_tensor.mean()}, std: {image_tensor.std()}")
            
            # Log histogram of pixel values
            hist = torch.histc(image_tensor, bins=10, min=0, max=1)
            logger.info(f"Histogram of pixel values: {hist}")

            # Generate latent
            latent = self.image_to_latent(image_tensor)

            return (image_tensor, {"samples": latent})

        except Exception as e:
            logger.error(f"Error in generate method: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def image_to_latent(self, image_tensor):
        # Convert image tensor to latent space
        latent = image_tensor * 2 - 1  # Scale to [-1, 1]
        latent = latent.permute(0, 2, 3, 1)  # [B, H, W, C]
        return latent

class LuminamGPTConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "LuminaWrapper"

    def convert(self, image):
        # Ensure the image is in the format ComfyUI expects
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] == 3:
            image = image.permute(0, 2, 3, 1)
        
        # Ensure values are in [0, 1] range
        image = image.clamp(0, 1)
        
        return (image,)

class LuminamGPTDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "LuminaWrapper"

    def decode(self, latent):
        # Extract the samples from the latent dict
        latent_samples = latent['samples']
        
        # Ensure the latent is in the correct shape [B, C, H, W]
        if latent_samples.dim() == 3:
            latent_samples = latent_samples.unsqueeze(0)
        if latent_samples.shape[1] != 3:
            latent_samples = latent_samples.permute(0, 3, 1, 2)
        
        # Convert from [-1, 1] to [0, 1] range
        image = (latent_samples + 1) / 2
        
        # Clamp values to ensure they're in [0, 1] range
        image = image.clamp(0, 1)
        
        # Convert from [B, C, H, W] to [B, H, W, C]
        image = image.permute(0, 2, 3, 1)
        
        return (image,)

NODE_CLASS_MAPPINGS = {
    "LuminamGPTLoader": LuminamGPTLoader,
    "LuminamGPTCropSelector": LuminamGPTCropSelector,
    "LuminamGPTImageGenerate": LuminamGPTImageGenerate,
    "LuminamGPTConverter": LuminamGPTConverter,
    "LuminamGPTDecoder": LuminamGPTDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminamGPTLoader": "Load Lumina-mGPT Model",
    "LuminamGPTCropSelector": "Lumina-mGPT Crop Selector",
    "LuminamGPTImageGenerate": "Lumina-mGPT Image Generate",
    "LuminamGPTConverter": "Lumina-mGPT Converter",
    "LuminamGPTDecoder": "Lumina-mGPT Decoder",
}