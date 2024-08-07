# ComfyUI-Lumina-mGPT-Wrapper

## Lumina-mGPT Node for ComfyUI

This custom node seamlessly integrates the Lumina-mGPT model into ComfyUI, enabling high-quality image generation using the advanced Lumina text-to-image pipeline. It offers a robust implementation with support for various model sizes and advanced features.

## Features

- Harnesses the power of the Lumina-mGPT model for state-of-the-art image generation
- Supports multiple model sizes: 512, 768, 768-Omni, and 1024
- Offers a range of generation parameters for fine-tuned control
- Implements Lumina-specific features including cfg-scale and image top-k sampling
- Automatic model downloading for seamless setup
- Outputs generated images directly

## Installation

1. Ensure you have ComfyUI installed and properly set up.
2. Clone this repository into your ComfyUI custom nodes directory:
3. The required dependencies will be automatically installed.

## Usage

1. Launch ComfyUI.
2. Locate the "Load Lumina-mGPT Model" node in the node selection menu.
3. Add the node to your workflow and connect it to a "Lumina-mGPT Image Generate" node.
4. Configure the node parameters as desired.
5. Execute your workflow to generate images.

## Parameters

### Load Lumina-mGPT Model
- `model`: Choose from available model sizes (512, 768, 768-Omni, 1024)
- `precision`: Select precision (bf16 or fp32)
- `target_size`: Set the target image size

### Lumina-mGPT Image Generate
- `prompt`: Text prompt for image generation
- `negative_prompt`: Negative text prompt
- `cfg`: Classifier-free guidance scale (default: 4.0)
- `seed`: Random seed for generation (-1 for random)
- `steps`: Number of inference steps (default: 30)
- `image_top_k`: Top-k sampling parameter for image generation (default: 2000)

## Outputs

- `IMAGE`: Generated image

## Known Features and Limitations

- Supports multiple model sizes for different use cases
- Implements cfg and image top-k parameters for controlling the generation process
- Currently outputs images directly; no need for additional VAE decoding

## Example Outputs

[Include some example images here]

## Troubleshooting

If you encounter any issues, please check the console output for error messages. Common issues include:

- Insufficient GPU memory
- Missing dependencies
- Incorrect model path

For further assistance, please open an issue on the GitHub repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Lumina-mGPT](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768) for the Lumina-mGPT model
- The ComfyUI community for their continuous support and inspiration
