# 🧹 Git Repository Cleanup Guide  
### Remove Large Files & Reduce `.git` Folder Size (Production Version)

This document explains how to clean an oversized `.git` folder caused by previously committed large files such as model weights, images, ZIP files, or virtual environments.  
Follow these steps to reduce a multi-GB repository to a clean, lightweight state.

```
A convenient cleanup script is available at:
bash scripts/cleanup_git.sh

*Make sure to run this from the project root.
Before running cleanup_git.sh,
ensure your working directory is completely clean
(git status must show 'nothing to commit').
```



---

# 1. Prepare `.gitignore` (Prevent Future Large Files)

Create or update `.gitignore`:

```
# Ignore model weights
assets/models/*
!assets/models/**/.gitkeep

# Ignore venv
venv/
*.pth
*.pt

# Ignore zip archives
*.zip

# Python cache
__pycache__/
*.pyc
```

Create `.gitkeep` placeholders:

```bash
touch assets/models/.gitkeep
touch assets/models/sam/.gitkeep
touch assets/models/yolo/.gitkeep
```

---

# 2. Stage the Current Cleaned State

```bash
git add .
git commit -m "chore: clean repo structure and update .gitignore"
```

This updates the working tree but **does not reduce `.git` size yet**.

---

# 3. Install `git-filter-repo`  
(Required to remove large files from history)

### Ubuntu (recommended)

```bash
sudo apt update
sudo apt install git-filter-repo -y
```

Check installation:

```bash
git filter-repo --help
```

---

# 4. Remove Large Files From Git History

Delete all blobs larger than **50 MB**:

```bash
git filter-repo --strip-blobs-bigger-than 50M
```

This rewrites your Git history and marks old objects as garbage,  
but the `.git` directory will **not shrink yet**.

---

# 5. Run Git Garbage Collection  
(Actually removes data from disk)

```bash
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

Now the `.git` folder will shrink from gigabytes → megabytes.

---

# 6. Force Push (If Using GitHub)

Because history has changed:

```bash
git push --force
```

---

# 7. Verify `.git` Size

```bash
du -h --max-depth=1 | sort -hr
```

or inspect `.git` specifically:

```bash
du -h .git --max-depth=1 | sort -hr
```

A normal `.git` size is **10–80MB**.

---

# 8. Optional: Check for Remaining Large Files

Find files larger than 50MB:

```bash
find . -type f -size +50M -exec du -h {} +
```

Check large blobs in Git history:

```bash
git rev-list --objects --all \
 | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
 | grep '^blob' \
 | sort -k3nr | head -20
```

---

# ✔ Summary (Quick Version)

```
# 1. Prepare ignore rules
edit .gitignore
touch assets/models/.gitkeep

# 2. Commit clean state
git add .
git commit -m "cleanup"

# 3. Remove large files from history
sudo apt install git-filter-repo -y
git filter-repo --strip-blobs-bigger-than 50M

# 4. Delete garbage
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Force push
git push --force

# 6. Check size
du -h --max-depth=1
```

---

# ✅ Notes

- This process **rewrites Git history**.  
- Safe for personal repositories; collaborative repos require coordination.  
- Once done, your project becomes lightweight & clean.