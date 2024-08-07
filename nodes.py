import torch
import os
import sys
import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths
from PIL import Image
import logging
import traceback
from .utils import download_and_load_model, CustomFlexARInferenceSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LuminamGPTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (["Alpha-VLLM/Lumina-mGPT-7B-512", "Alpha-VLLM/Lumina-mGPT-7B-768", "Alpha-VLLM/Lumina-mGPT-7B-768-Omni", "Alpha-VLLM/Lumina-mGPT-7B-1024"], {"default": "Alpha-VLLM/Lumina-mGPT-7B-768"}),
            "precision": (["bf16", "fp32"], {"default": "bf16"}),
            "target_size": ("INT", {"default": 768, "min": 512, "max": 1024, "step": 256}),
        }}

    RETURN_TYPES = ("LUMINAMGPT",)
    RETURN_NAMES = ("lumina_mgpt_model",)
    FUNCTION = "load_model"
    CATEGORY = "LuminaWrapper"

    def load_model(self, model, precision, target_size):
        try:
            model_path = os.path.join(folder_paths.models_dir, "lumina_mgpt", model.split("/")[-1])
            tokenizer_path = os.path.join(model_path, 'ckpts', 'chameleon', 'tokenizer')
            
            logger.info(f"Loading model with precision={precision}, target_size={target_size}")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Tokenizer path: {tokenizer_path}")
            
            inference_solver = download_and_load_model(model, precision, target_size)
            logger.info(f"Successfully loaded model: {model}")
            return (inference_solver,)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise

class LuminamGPTImageGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lumina_mgpt_model": ("LUMINAMGPT",),
                "prompt": ("STRING", {"multiline": True, "default": "Generate an image of a dog playing in water, with a waterfall in the background."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
                "image_top_k": ("INT", {"default": 2000, "min": 1, "max": 10000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "LuminaWrapper"

    def generate(self, lumina_mgpt_model, prompt, negative_prompt, cfg, seed, steps, image_top_k):
        try:
            q1 = f"Generate an image according to the following prompt:\n{prompt}"
            if negative_prompt:
                q1 += f"\nDo not include: {negative_prompt}"

            logger.info(f"Creating logits processor with cfg={cfg}, image_top_k={image_top_k}")
            logits_processor = lumina_mgpt_model.create_logits_processor(cfg=cfg, image_top_k=image_top_k)

            logger.info(f"Starting image generation with seed={seed}, steps={steps}")
            generated = lumina_mgpt_model.generate(
                images=[],
                qas=[[q1, None]],
                max_gen_len=8192,
                temperature=1.0,
                logits_processor=logits_processor,
                num_inference_steps=steps,
                seed=seed if seed != -1 else None
            )

            logger.info("Image generation completed")
            a1, new_image = generated[0], generated[1][0]
            
            if new_image is None:
                logger.error("Generated image is None")
                raise ValueError("Failed to generate image")

            new_image_tensor = torch.tensor(new_image).permute(2, 0, 1).float() / 255.0
            
            logger.info("Successfully generated image")
            return (new_image_tensor,)
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

NODE_CLASS_MAPPINGS = {
    "LuminamGPTLoader": LuminamGPTLoader,
    "LuminamGPTImageGenerate": LuminamGPTImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminamGPTLoader": "Load Lumina-mGPT Model",
    "LuminamGPTImageGenerate": "Lumina-mGPT Image Generate",
}