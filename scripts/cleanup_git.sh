#!/bin/bash

echo "===== Git Repository Cleanup Script ====="
echo "This script will remove large files from Git history and shrink the .git directory."
echo

# 1. Check for git-filter-repo
if ! command -v git-filter-repo &> /dev/null; then
    echo "[INFO] git-filter-repo not found. Installing..."
    sudo apt update
    sudo apt install git-filter-repo -y
else
    echo "[INFO] git-filter-repo already installed."
fi

# 2. Remove blobs > 50MB
echo
echo "[STEP] Removing blobs larger than 50MB..."
git filter-repo --strip-blobs-bigger-than 50M

# 3. Git reflog cleanup
echo
echo "[STEP] Cleaning reflog..."
git reflog expire --expire=now --all

# 4. Git garbage collection
echo
echo "[STEP] Running aggressive git gc..."
git gc --prune=now --aggressive

# 5. Show result
echo
echo "===== Cleanup Complete! ====="
echo "Current repository size:"
du -h --max-depth=1 | sort -hr
echo
echo "If you are using GitHub, run:"
echo "  git push --force"
echo "========================================="