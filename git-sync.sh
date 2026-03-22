#!/bin/bash

echo "--- Git Auto-Commit Script ---"

# 1. Check requirements
read -p "1. requirements.txt is set? (y/n): " req_ans
if [[ "$req_ans" != "y" ]]; then
    echo "Process aborted. Please update your requirements.txt."
    echo "Tip: You can generate it quickly by running: uv pip freeze > requirements.txt"
    exit 1
fi

# 2. Git Add
read -p "2. Run 'git add .'? (y/n): " add_ans
if [[ "$add_ans" == "y" ]]; then
    git add .
    echo "✅ Files staged."
    git status -s
else
    echo "Process aborted. You can stage files manually."
    exit 1
fi

# 3. Git Commit
read -p "3. Enter your commit message: " commit_msg
if [[ -n "$commit_msg" ]]; then
    git commit -m "$commit_msg"
    echo "✅ Commit successful."
else
    echo "Process aborted. Commit message cannot be empty."
    exit 1
fi

# 4. Git Push
read -p "4. Run 'git push'? (y/n): " push_ans
if [[ "$push_ans" == "y" ]]; then
    git push
    echo "✅ Code pushed successfully!"
else
    echo "Skipping push. Your commits are saved locally."
fi

echo "--- Done ---"