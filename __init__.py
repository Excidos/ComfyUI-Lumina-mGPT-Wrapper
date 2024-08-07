import os
import sys
import subprocess

def run_install_script():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    install_script = os.path.join(current_dir, "install.py")
   
    print(f"Running Lumina-mGPT installation script from {install_script}")
   
    try:
        result = subprocess.run([sys.executable, install_script], check=True, capture_output=True, text=True)
        print(result.stdout)
        print("Lumina-mGPT installation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during Lumina-mGPT installation: {e}")
        print(f"Error output:\n{e.stdout}\n{e.stderr}")
        print("The custom node may not function correctly. Please check the error and try manual installation.")
        return False

def check_lumina_mgpt_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lumina_mgpt_path = os.path.join(current_dir, 'Lumina-mGPT')
    required_files = [
        os.path.join(lumina_mgpt_path, 'lumina_mgpt', 'ckpts', 'chameleon', 'tokenizer', 'text_tokenizer.json'),
        os.path.join(lumina_mgpt_path, 'lumina_mgpt', 'ckpts', 'chameleon', 'tokenizer', 'vqgan.yaml'),
        os.path.join(lumina_mgpt_path, 'lumina_mgpt', 'ckpts', 'chameleon', 'tokenizer', 'vqgan.ckpt')
    ]
    return all(os.path.exists(file) for file in required_files)

def import_node_classes():
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("Successfully imported Lumina-mGPT node classes")
        return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except ImportError as e:
        print(f"Error importing Lumina-mGPT node classes: {e}")
        print("The custom node may not be available. Please check the installation and try again.")
        return {}, {}

# Check if files exist, if not, run the installation script
if not check_lumina_mgpt_files():
    print("Lumina-mGPT files not found. Running the installation script.")
    installation_successful = run_install_script()
else:
    print("Lumina-mGPT files found. Proceeding with initialization.")
    installation_successful = True

# After installation or if files were already present, attempt to import the node classes
if installation_successful:
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = import_node_classes()
else:
    print("Skipping node class import due to installation failure.")
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("Lumina-mGPT wrapper initialization complete")