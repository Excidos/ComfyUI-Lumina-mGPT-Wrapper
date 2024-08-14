# ComfyUI-Lumina-mGPT-Wrapper

## Overview
This custom node integrates the Lumina-mGPT model into ComfyUI, enabling high-quality image generation using the advanced Lumina text-to-image pipeline. It offers a robust implementation with support for various model sizes and advanced features.

## Features
- Harnesses the power of the Lumina-mGPT model for state-of-the-art image generation
- Supports multiple model sizes: 512, 768, 768-Omni, and 1024
- Offers a range of generation parameters for fine-tuned control
- Implements Lumina-specific features including cfg-scale and image top-k sampling
- Automatic model downloading for seamless setup
- Outputs both generated images and latent representations
- Includes a converter node for ComfyUI compatibility
- Provides a decoder node for latent-to-image conversion

## Preparation
Since the Chameleon implementation in transformers does not contain the VQ-VAE decoder, please manually download the original VQ-VAE weights provided by Meta and place them in the following directory:

### NOTE: I've added the files to the respective path however they will need to be extracted from the zip file before running the node.

```
Lumina-mGPT
- lumina_mgpt/
    - ckpts/
        - chameleon/
            - tokenizer/
                - text_tokenizer.json
                - vqgan.yaml
                - vqgan.ckpt
- xllmx/
- ...
```

You can download the required files from [Meta's Chameleon Downloads](https://ai.meta.com/resources/models-and-libraries/chameleon-downloads/).

## Installation
1. Ensure you have ComfyUI installed and properly set up.
2. Clone this repository into your ComfyUI custom nodes directory:
   ```
   git clone https://github.com/Excidos/ComfyUI-Lumina-mGPT-Wrapper.git
   ```
3. The required dependencies will be automatically installed.

4. If you are having trouble you may have to install XLLMX independently but copying the path of your python_embedded python.exe and running in the Lumina-mGPT
   directory

    ### go to the root path of the project
    cd Lumina_mGPT
    ### install as package
    "Path to your embedded python.exe" pip -m install -e .

## Usage
1. Launch ComfyUI.
2. Locate the "Load Lumina-mGPT Model" node in the node selection menu.
3. Add the node to your workflow and connect it to a "Lumina-mGPT Image Generate" node.
4. (Optional) Use the "Lumina-mGPT Crop Selector" to choose a specific resolution.
5. Configure the node parameters as desired.
6. Connect the output to either a "Lumina-mGPT Converter" or "Lumina-mGPT Decoder" node for further processing or display.
7. Execute your workflow to generate images.

## Nodes and Parameters

### Load Lumina-mGPT Model
- `model`: Choose from available model sizes (512, 768, 768-Omni, 1024)
- `precision`: Select precision (bf16 or fp32)

### Lumina-mGPT Crop Selector
- `target_size`: Select from 512, 768, or 1024
- `aspectRatio`: Choose from various aspect ratios (1:1, 4:3, 16:9, etc.)

### Lumina-mGPT Image Generate
- `lumina_mgpt_model`: Connected from the Load Lumina-mGPT Model node
- `prompt`: Text prompt for image generation
- `resolution`: Image resolution (can be connected from Crop Selector)
- `cfg`: Classifier-free guidance scale
- `seed`: Random seed for generation (0 for random)
- `image_top_k`: Top-k sampling parameter for image generation
- `temperature`: Controls randomness in generation

### Lumina-mGPT Converter
- `image`: Input image to convert to ComfyUI-compatible format

### Lumina-mGPT Decoder
- `latent`: Input latent representation to decode into an image

## Outputs
- `IMAGE`: Generated or decoded image
- `LATENT`: Latent representation of the generated image

## Examples

![Screenshot 2024-08-14 082926](https://github.com/user-attachments/assets/c6c064ad-5805-4d71-9d90-cb665728a995)


## Known Features and Limitations
- Supports multiple model sizes for different use cases
- Implements cfg and image top-k parameters for controlling the generation process
- Outputs both images and latent representations
- Includes converter and decoder nodes for enhanced compatibility and flexibility

## Troubleshooting
If you encounter any issues, please check the console output for error messages. Common issues include:
- Insufficient GPU memory
- Missing dependencies
- Incorrect model or tokenizer path

For further assistance, please open an issue on the GitHub repository.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements
- [Lumina-mGPT](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768) for the Lumina-mGPT model
- The ComfyUI community for their continuous support and inspiration
- Meta for providing the Chameleon VQ-VAE weights
