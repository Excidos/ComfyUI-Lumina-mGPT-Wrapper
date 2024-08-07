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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lumina_mgpt_path = os.path.join(current_dir, 'Lumina-mGPT')
    modified_requirements_path = os.path.join(current_dir, 'modified_requirements.txt')
    
    print(f"Current directory: {current_dir}")
    print(f"Lumina_mGPT path: {lumina_mgpt_path}")
    
    # Check if git is installed
    if not run_command("git --version"):
        print("Git is not installed or not in PATH. Please install Git and try again.")
        return False

    # Clone Lumina_mGPT if it doesn't exist in the correct location
    if not os.path.exists(lumina_mgpt_path):
        print(f"Cloning Lumina_mGPT repository to: {lumina_mgpt_path}")
        if not run_command(f'git clone "https://github.com/Alpha-VLLM/Lumina-mGPT.git" "{lumina_mgpt_path}"'):
            print("Failed to clone the repository. Please check your internet connection and try again.")
            return False
    else:
        print("Lumina_mGPT repository already exists. Updating...")
        if not run_command("git pull", cwd=lumina_mgpt_path):
            print("Failed to update the repository. Continuing with existing files.")

    # Verify the repository was cloned successfully
    if not os.path.exists(lumina_mgpt_path):
        print(f"Error: Failed to clone Lumina_mGPT repository to {lumina_mgpt_path}")
        return False

    # Install modified Lumina_mGPT requirements
    print("Installing modified Lumina_mGPT requirements...")
    if not os.path.exists(modified_requirements_path):
        print(f"Error: modified_requirements.txt not found at {modified_requirements_path}")
        return False
    if not run_command(f"\"{sys.executable}\" -m pip install -r \"{modified_requirements_path}\""):
        print("Failed to install requirements. Please check the error messages above.")
        return False

    # Install xllmx from Lumina_mGPT
    print("Installing xllmx...")
    if not run_command(f"\"{sys.executable}\" -m pip install -e \"{lumina_mgpt_path}\""):
        print("Failed to install xllmx. Please check the error messages above.")
        return False

    # Download necessary files
    print("Downloading necessary files...")
    chameleon_path = os.path.join(lumina_mgpt_path, "lumina_mgpt", "ckpts", "chameleon", "tokenizer")
    os.makedirs(chameleon_path, exist_ok=True)
    
    files_to_download = [
        ("https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768/resolve/main/tokenizer/text_tokenizer.json", "text_tokenizer.json"),
        ("https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768/resolve/main/tokenizer/vqgan.yaml", "vqgan.yaml"),
        ("https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768/resolve/main/tokenizer/vqgan.ckpt", "vqgan.ckpt")
    ]

    for url, filename in files_to_download:
        file_path = os.path.join(chameleon_path, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            if not run_command(f"curl -L {url} -o \"{file_path}\""):
                print(f"Failed to download {filename}. Please download it manually and place it in {chameleon_path}")
        else:
            print(f"{filename} already exists. Skipping download.")

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