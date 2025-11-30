import os
import urllib.request

# ---------------------------
# 다운로드 목록 정의
# ---------------------------

DOWNLOAD_TARGETS = [
    {
        "name": "SAM ViT-H",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "path": "assets/models/sam/sam_vit_h_4b8939.pth",
        "size": "≈ 2.4GB"
    },
    {
        "name": "YOLOv8n",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "path": "assets/models/yolo/yolov8n.pt",
        "size": "≈ 6MB"
    }
]


# ---------------------------
# 유틸 함수
# ---------------------------

def ensure_dir(filepath):
    """해당 파일이 저장될 디렉토리를 자동 생성"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def download_file(url, path):
    """파일이 없으면 다운로드, 있으면 스킵"""
    if os.path.exists(path):
        print(f"[SKIP] 이미 있음: {path}")
        return

    print(f"[DOWNLOAD] {url}")
    print(f" → 저장: {path}")

    ensure_dir(path)
    urllib.request.urlretrieve(url, path)
    print(f"[DONE] 다운로드 완료: {path}\n")


# ---------------------------
# 메인
# ---------------------------

def main():
    print("===========================================")
    print("   📦 Weight Downloader (SAM + YOLO)       ")
    print("===========================================\n")

    for item in DOWNLOAD_TARGETS:
        print(f"== {item['name']} ({item['size']}) ==")
        download_file(item["url"], item["path"])

    print("\n모든 모델 다운로드 완료!")


if __name__ == "__main__":
    main()