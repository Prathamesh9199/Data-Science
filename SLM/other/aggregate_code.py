import argparse
from pathlib import Path

def aggregate_python_files(root_folder: str, output_file: str):
    root_path = Path(root_folder)
    
    # 1. Validate the directory
    if not root_path.exists() or not root_path.is_dir():
        print(f"❌ Error: The folder '{root_folder}' does not exist.")
        return

    # Folders we usually want to ignore to avoid bloating the text file
    ignore_folders = {'.venv', 'venv', 'env', '__pycache__', '.git', '.tox'}

    processed_count = 0

    print(f"🔍 Scanning '{root_path.absolute()}' for Python files...")

    # 2. Open the output file and start writing
    with open(output_file, 'w', encoding='utf-8') as out_f:
        
        # rglob('*.py') recursively finds all .py files
        for file_path in root_path.rglob('*.py'):
            
            # Skip files that are inside ignored directories
            if any(ignored in file_path.parts for ignored in ignore_folders):
                continue
                
            try:
                # Get a clean relative path (e.g., folder/script.py instead of C:/.../folder/script.py)
                try:
                    display_path = file_path.relative_to(root_path)
                except ValueError:
                    display_path = file_path

                # Read the actual code
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    code = in_f.read()
                
                # Write to the text file in your exact requested format
                out_f.write(f"{display_path}\n")
                out_f.write(f"{code}\n")
                out_f.write("=======================================\n\n")
                
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️ Skipping {file_path} due to error: {e}")

    if processed_count > 0:
        print(f"✅ Success! Compiled {processed_count} files into '{output_file}'.")
    else:
        print("🤷 No valid .py files were found in the specified directory.")

if __name__ == "__main__":
    # Set up command line arguments for easy use

    root_folder = '/home/prathamesh/Data-Science/SLM/'
    output = '/home/prathamesh/Data-Science/SLM/code.txt'

    aggregate_python_files(root_folder, output)