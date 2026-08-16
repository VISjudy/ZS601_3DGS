# 打包代码 zip（排除 build / __pycache__ / egg-info / .git 等），供上传 Google Drive 后在 Colab 使用
import os
import zipfile

SRC = r"d:\hhs\gaussian-splattingWithMask-0811\gaussian-splattingWithMask"
DST = r"d:\hhs\gaussian-splattingWithMask-0811\gs_code.zip"

EXCLUDE_DIRS = {"build", "__pycache__", ".ipynb_checkpoints", ".git",
                "diff_gaussian_rasterization.egg-info", "fused_ssim.egg-info",
                "simple_knn.egg-info", "fused-ssim000"}
EXCLUDE_EXTS = {".pyc", ".obj", ".lib", ".exp", ".pyd", ".so"}

count = 0
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() in EXCLUDE_EXTS:
                continue
            full = os.path.join(root, f)
            arc = os.path.join("gaussian-splattingWithMask", os.path.relpath(full, SRC))
            zf.write(full, arc)
            count += 1
print("打包完成：{} 个文件 -> {}".format(count, DST))
print("大小：{:.1f} MB".format(os.path.getsize(DST) / 1024 / 1024))
