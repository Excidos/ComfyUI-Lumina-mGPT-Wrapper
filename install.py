import subprocess
import sys
import os
import shutil

def run_command(command, cwd=None):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True, cwd=cwd)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_requirements():
    # Determine the path to python_embedded
    comfyui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    python_embedded_path = os.path.join(comfyui_path, '..', 'python_embeded', 'python.exe')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lumina_mgpt_path = os.path.join(current_dir, 'Lumina-mGPT')
    modified_requirements_path = os.path.join(current_dir, 'modified_requirements.txt')
    
    print(f"Current directory: {current_dir}")
    print(f"Lumina_mGPT path: {lumina_mgpt_path}")
    print(f"Python embedded path: {python_embedded_path}")
    
    # Verify the Lumina-mGPT directory exists
    if not os.path.exists(lumina_mgpt_path):
        print(f"Error: Lumina-mGPT directory not found at {lumina_mgpt_path}")
        return False

    # Install modified Lumina_mGPT requirements
    print("Installing modified Lumina_mGPT requirements...")
    if not os.path.exists(modified_requirements_path):
        print(f"Error: modified_requirements.txt not found at {modified_requirements_path}")
        return False
    if not run_command(f"\"{python_embedded_path}\" -m pip install -r \"{modified_requirements_path}\""):
        print("Failed to install requirements. Please check the error messages above.")
        return False

    # Install xllmx from Lumina_mGPT
    print("Installing xllmx...")
    if not run_command(f"\"{python_embedded_path}\" -m pip install -e \"{lumina_mgpt_path}\""):
        print("Failed to install xllmx. Please check the error messages above.")
        return False

    # Check for necessary files
    print("Checking for necessary files...")
    chameleon_path = os.path.join(lumina_mgpt_path, "lumina_mgpt", "ckpts", "chameleon", "tokenizer")
    files_to_check = ["text_tokenizer.json", "vqgan.yaml", "vqgan.ckpt"]

    for filename in files_to_check:
        file_path = os.path.join(chameleon_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found at {file_path}")
            print(f"Please ensure {filename} is present in the {chameleon_path} directory.")
        else:
            print(f"{filename} found at {file_path}")

    print("Installation completed successfully!")
    return True

if __name__ == "__main__":
    success = install_requirements()
    if not success:
        print("Installation failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("Installation completed successfully.")
        sys.exit(0)