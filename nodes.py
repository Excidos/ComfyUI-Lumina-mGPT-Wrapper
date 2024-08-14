import os
import logging
import torch
import numpy as np
from PIL import Image
from lumina_mgpt.inference_solver import FlexARInferenceSolver
from lumina_mgpt.data.item_processor import FlexARItemProcessor, generate_crop_size_list
from huggingface_hub import snapshot_download

import folder_paths

logger = logging.getLogger(__name__)

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "Lumina-mGPT", "lumina_mgpt", "ckpts", "chameleon", "tokenizer")

# Define valid crop sizes for different model versions
VALID_CROP_SIZES = {
    512: [
        "1024x256", "992x256", "960x256", "928x256", "896x256", "896x288",
        "864x288", "832x288", "800x288", "800x320", "768x320", "736x320",
        "736x352", "704x352", "672x352", "672x384", "640x384", "608x384",
        "608x416", "576x416", "576x448", "544x448", "544x480", "512x480",
        "512x512", "480x512", "480x544", "448x544", "448x576", "416x576",
        "416x608", "384x608", "384x640", "384x672", "352x672", "352x704",
        "352x736", "320x736", "320x768", "320x800", "288x800", "288x832",
        "288x864", "288x896", "256x896", "256x928", "256x960", "256x992",
        "256x1024"
    ],
    768: [
        "1536x384", "1504x384", "1472x384", "1440x384", "1408x384", "1408x416",
        "1376x416", "1344x416", "1312x416", "1312x448", "1280x448", "1248x448",
        "1216x448", "1216x480", "1184x480", "1152x480", "1152x512", "1120x512",
        "1088x512", "1056x512", "1056x544", "1024x544", "1024x576", "992x576",
        "960x576", "960x608", "928x608", "896x608", "896x640", "864x640",
        "864x672", "832x672", "832x704", "800x704", "800x736", "768x736",
        "768x768", "736x768", "736x800", "704x800", "704x832", "672x832",
        "672x864", "640x864", "640x896", "608x896", "608x928", "608x960",
        "576x960", "576x992", "576x1024", "544x1024", "544x1056", "512x1056",
        "512x1088", "512x1120", "512x1152", "480x1152", "480x1184", "480x1216",
        "448x1216", "448x1248", "448x1280", "448x1312", "416x1312", "416x1344",
        "416x1376", "416x1408", "384x1408", "384x1440", "384x1472", "384x1504",
        "384x1536"
    ]
}

class LuminamGPTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (["Alpha-VLLM/Lumina-mGPT-7B-512", "Alpha-VLLM/Lumina-mGPT-7B-768"], {"default": "Alpha-VLLM/Lumina-mGPT-7B-768"}),
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
                logger.info(f"Model directory not found: {model_dir}. Downloading from Hugging Face.")
                snapshot_download(repo_id=model, local_dir=model_dir, local_dir_use_symlinks=False)

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
                "target_size": (["512", "768"], {"default": "768"}),
                "aspectRatio": ([
                    "1:1 - square",
                    "4:3 - standard",
                    "16:9 - widescreen",
                    "2:3 - portrait",
                    "3:2 - landscape"
                ],),
                "crop_size": (["None"] + VALID_CROP_SIZES[512] + VALID_CROP_SIZES[768], {"default": "None"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("resolution",)
    FUNCTION = "select_crop"
    CATEGORY = "LuminaWrapper"

    def select_crop(self, target_size, aspectRatio, crop_size):
        target_size = int(target_size)
        if crop_size != "None":
            resolution = crop_size
        else:
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

            # Find the closest valid crop size
            closest_resolution = min(VALID_CROP_SIZES[target_size], key=lambda x: abs(int(x.split('x')[0]) - width) + abs(int(x.split('x')[1]) - height))

            if resolution != closest_resolution:
                logger.warning(f"Adjusted resolution from {resolution} to {closest_resolution} to match valid crop sizes for {target_size} model.")
                resolution = closest_resolution

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
                "resolution": ("STRING", {"default": "768x768"}),
                "resolution_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "LuminaWrapper"

    def generate(self, lumina_mgpt_model, prompt, cfg, seed, image_top_k, temperature, resolution="768x768", resolution_input=None):
        try:
            # Use resolution_input if provided, otherwise use resolution
            resolution_to_use = resolution_input if resolution_input else resolution
            width, height = map(int, resolution_to_use.split('x'))

            # Determine the target size based on the resolution
            target_size = 768 if max(width, height) > 512 else 512

            # Ensure resolution is valid
            if resolution_to_use not in VALID_CROP_SIZES[target_size]:
                closest_resolution = min(VALID_CROP_SIZES[target_size], key=lambda x: abs(int(x.split('x')[0]) - width) + abs(int(x.split('x')[1]) - height))
                logger.warning(f"Adjusted resolution from {resolution_to_use} to {closest_resolution} to match valid crop sizes for {target_size} model.")
                resolution_to_use = closest_resolution
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
            
            # Generate latent
            latent = self.image_to_latent(image_tensor)

            return (image_tensor, latent)

        except Exception as e:
            logger.error(f"Error in generate method: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def image_to_latent(self, image_tensor):
        # Convert image tensor to latent space
        latent = image_tensor * 2 - 1  # Scale to [-1, 1]
        
        # Ensure the latent is in the shape [B, C, H, W]
        if latent.shape[1] != 4:
            # If we have 3 channels, we can repeat one channel to get 4
            latent = latent.repeat(1, 2, 1, 1)[:, :4, :, :]
        
        return {"samples": latent}

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
                "output_type": (["IMAGE", "LATENT"], {"default": "IMAGE"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "decode"
    CATEGORY = "LuminaWrapper"

    def decode(self, latent, output_type):
        # Extract the samples from the latent dict
        latent_samples = latent['samples']
        
        # Ensure the latent is in the correct shape [B, C, H, W]
        if latent_samples.dim() == 3:
            latent_samples = latent_samples.unsqueeze(0)
        
        # Convert from [-1, 1] to [0, 1] range
        normalized_samples = (latent_samples + 1) / 2
        normalized_samples = normalized_samples.clamp(0, 1)

        if output_type == "IMAGE":
            # For image output, ensure we have 3 channels (RGB)
            if normalized_samples.shape[1] != 3:
                normalized_samples = normalized_samples[:, :3, :, :]
            
            # Convert from [B, C, H, W] to [B, H, W, C]
            image = normalized_samples.permute(0, 2, 3, 1)
            return (image, None)
        
        else:  # LATENT output
            # For latent output, ensure we have 4 channels
            B, C, H, W = normalized_samples.shape
            if C != 4:
                normalized_samples = normalized_samples.repeat(1, 2, 1, 1)[:, :4, :, :]
            
            # Adjust the spatial dimensions to match ComfyUI expectations (assuming 8x downscaling)
            target_h, target_w = H // 8, W // 8
            comfy_latent = torch.nn.functional.interpolate(normalized_samples, size=(target_h, target_w), mode='bicubic')
            
            # Scale back to [-1, 1] range
            comfy_latent = comfy_latent * 2 - 1
            
            return (None, {"samples": comfy_latent})

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
