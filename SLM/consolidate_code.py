import os
from config import SLMConfig

cnfg = SLMConfig()

SKIP_DIRS = {".venv", "venv", "env", "__pycache__", ".git", 
             "node_modules", "site-packages", ".mypy_cache", ".tox", 
             "base_models", "batches", "checkpoints", "data", "data copy",
             "data_bkp", "other", "plan", "training_artifacts"
            }

def consolidate_py_files(root_folder, output_file="consolidated.txt"):
    with open(output_file, "w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(root_folder):
            # Prune skip dirs in-place so os.walk won't descend into them
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                if filename.endswith(".py"):
                    filepath = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(filepath, root_folder)
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        if cnfg.hf_token in code:
                            code = code.replace(cnfg.hf_token, '<HF_TOKEN>')
                    out.write(f"<<{rel_path}>>\n")
                    out.write(code)
                    out.write("\n========================\n")
    print(f"Done! Output written to: {output_file}")

if __name__ == "__main__":
    root = "/home/prathamesh/Data-Science/SLM/"
    output = "codebase.txt"
    consolidate_py_files(root, output)