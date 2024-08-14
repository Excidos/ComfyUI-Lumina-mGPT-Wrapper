import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add Lumina-mGPT paths to the Python path
lumina_mgpt_path = os.path.join(current_dir, 'Lumina-mGPT')
lumina_mgpt_main_path = os.path.join(lumina_mgpt_path, 'lumina_mgpt')
if lumina_mgpt_path not in sys.path:
    sys.path.insert(0, lumina_mgpt_path)
if lumina_mgpt_main_path not in sys.path:
    sys.path.insert(0, lumina_mgpt_main_path)

logger.info(f"Current sys.path: {sys.path}")

TOKENIZER_PATH = os.path.join(current_dir, 'Lumina-mGPT', 'lumina_mgpt', 'ckpts', 'chameleon', 'tokenizer')
text_tokenizer_path = os.path.join(TOKENIZER_PATH, 'text_tokenizer.json')
if not os.path.exists(text_tokenizer_path):
    logger.error(f"text_tokenizer.json not found at {text_tokenizer_path}")
else:
    logger.info(f"text_tokenizer.json found at {text_tokenizer_path}")


def import_lumina_mgpt():
    try:
        global FlexARInferenceSolver, FlexARItemProcessor, ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration

        from lumina_mgpt.inference_solver import FlexARInferenceSolver
        from lumina_mgpt.data.item_processor import FlexARItemProcessor
        from lumina_mgpt.model.configuration_xllmx_chameleon import ChameleonXLLMXConfig
        from lumina_mgpt.model.modeling_xllmx_chameleon import ChameleonXLLMXForConditionalGeneration

        # Verify xllmx import
        import xllmx
        logger.info(f"xllmx module imported successfully from: {xllmx.__file__}")

        logger.info("Successfully imported Lumina-mGPT modules including xllmx")
        return True
    except ImportError as e:
        logger.error(f"Error importing Lumina-mGPT modules: {e}")
        logger.error(f"Current sys.path: {sys.path}")
        return False

def initialize_lumina_mgpt():
    if import_lumina_mgpt():
        try:
            from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
            return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        except Exception as e:
            logger.error(f"Error during node class import: {e}")
            return {}, {}
    else:
        logger.warning("Failed to import Lumina-mGPT modules. The custom node may not function correctly.")
        return {}, {}

# Initialize the Lumina-mGPT wrapper
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = initialize_lumina_mgpt()

# Export the necessary components
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'TOKENIZER_PATH']

logger.info("Lumina-mGPT wrapper initialization complete")