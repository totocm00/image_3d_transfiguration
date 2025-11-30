# 🧹 Git 저장소 용량 정리 가이드  
### 대용량 파일 제거 & `.git` 폴더 축소 (배포용 문서)

`.git` 폴더가 갑자기 GB가 되는 가장 흔한 이유는  
**가중치 파일(.pth), .pt, 이미지, ZIP, venv 같은 대용량 바이너리를 커밋한 적이 있기 때문**입니다.

이 문서는 이를 완전히 제거하여 저장소 전체를 가볍게 만드는 절차를 안내합니다.

```
scripts/cleanup_git.sh 간편 스크립트도 있습니다.  
bash scripts/cleanup_git.sh

*실행 전 반드시 프로젝트 루트에서 실행
cleanup_git.sh 실행 전,
반드시 작업 디렉토리가 완전히 깨끗한 상태인지
(git status가 'nothing to commit') 확인하세요.
```

---

# 1. .gitignore 구성 (대용량 파일 추적 방지)

`.gitignore`에 아래 내용을 추가하거나 유지합니다:

```
# 모델 가중치 무시
assets/models/*
!assets/models/**/.gitkeep

# 가상환경 무시
venv/

# 대용량 파일
*.pth
*.pt
*.zip

# 파이썬 캐시
__pycache__/
*.pyc
```

빈 폴더를 유지하기 위한 `.gitkeep` 생성:

```bash
touch assets/models/.gitkeep
touch assets/models/sam/.gitkeep
touch assets/models/yolo/.gitkeep
```

---

# 2. 현재 깨끗한 상태 커밋

```bash
git add .
git commit -m "chore: clean repo structure and update .gitignore"
```

> ❗ 이 단계는 현재 파일 상태를 정리할 뿐, `.git` 자체 용량은 줄어들지 않습니다.

---

# 3. git-filter-repo 설치  
(깃 히스토리 청소를 위해 꼭 필요함)

### Ubuntu 설치

```bash
sudo apt update
sudo apt install git-filter-repo -y
```

설치 확인:

```bash
git filter-repo --help
```

---

# 4. Git 히스토리에서 대용량 파일 제거

50MB 초과 파일을 히스토리에서 완전히 제거:

```bash
git filter-repo --strip-blobs-bigger-than 50M
```

> 이 과정은 히스토리를 재작성(rewrite)합니다.  
> `.git` 폴더는 아직 줄어들지 않습니다. (GC 필요)

---

# 5. Git 쓰레기(Garbage) 수집 & 디스크에서 완전 삭제  
**이 단계에서 실제 용량이 감소합니다.**

### 5-1. reflog 제거

```bash
git reflog expire --expire=now --all
```

### 5-2. aggressive GC 실행

```bash
git gc --prune=now --aggressive
```

이제 `.git` 폴더가 **수 GB → 수십 MB**로 줄어들어야 합니다.

---

# 6. GitHub로 강제 푸시 (히스토리 변경이므로)

```bash
git push --force
```

---

# 7. 용량 확인

```bash
du -h --max-depth=1 | sort -hr
```

또는 `.git` 내부 확인:

```bash
du -h .git --max-depth=1 | sort -hr
```

정상 `.git` 용량은 **10–80MB 범위**입니다.

---

# 8. 잔여 대용량 파일 검사 (선택)

50MB 이상 파일 검색:

```bash
find . -type f -size +50M -exec du -h {} +
```

Git 히스토리 내 blob 검사:

```bash
git rev-list --objects --all \
 | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
 | grep '^blob' | sort -k3nr | head -20
```

---

# ✔ 전체 요약 (Quick Summary)

```
1. .gitignore 구성 & .gitkeep 생성
2. git add . && git commit
3. git filter-repo --strip-blobs-bigger-than 50M
4. git reflog expire --expire=now --all
5. git gc --prune=now --aggressive
6. git push --force
7. 용량 확인
```

---

# 📌 참고

- 이 과정은 **Git 히스토리를 재작성**하므로 개인 리포는 문제 없음.  
- 협업 리포는 모든 팀원이 rebase 또는 재클론 필요.