# 打包 v2 代码 zip（供上传 Google Drive 后在 Colab 使用）
# 排除规则（已与团队经验同步，修改前请同步更新 AGENTS.md 第 7 节）：
# 1. 目录排除：构建缓存 / checkpoint 备份 / fused-ssim（Colab 用 git clone 安装，失败自动回退普通 ssim）
# 2. 文件排除：编译产物、训练日志 csv、文件名含非 ASCII 的备份参考文件
# 3. third_party 下的图片资源排除（glm 文档图片等，不影响编译）
import os
import re
import zipfile

SRC = r"d:\hhs\gaussian-splattingWithMask-0811\gaussian-splattingWithMask_v2"
DST = r"d:\hhs\gaussian-splattingWithMask-0811\gs_code_v2.zip"
TOP = "gaussian-splattingWithMask_v2"

EXCLUDE_DIRS = {"build", "__pycache__", ".ipynb_checkpoints", ".git", "doc",
                "diff_gaussian_rasterization.egg-info", "fused_ssim.egg-info",
                "simple_knn.egg-info", "fused-ssim000", "fused-ssim"}
EXCLUDE_EXTS = {".pyc", ".obj", ".lib", ".exp", ".pyd", ".so"}
EXCLUDE_FILES = {"loss_log.csv", "val_metrics.csv"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".gif", ".pdf"}
NON_ASCII = re.compile(r"[^\x00-\x7f]")

if os.path.exists(DST):
    os.remove(DST)

count = 0
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]
        in_third_party = "third_party" in root
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXCLUDE_EXTS or f in EXCLUDE_FILES:
                continue
            if NON_ASCII.search(f):          # 文件名含中文的备份参考文件不参与分发
                continue
            if in_third_party and ext in IMG_EXTS:
                continue
            full = os.path.join(root, f)
            arc = os.path.join(TOP, os.path.relpath(full, SRC)).replace(os.sep, "/")
            zf.write(full, arc)
            count += 1
print("打包完成：{} 个文件 -> {}".format(count, DST))
print("大小：{:.2f} MB".format(os.path.getsize(DST) / 1024 / 1024))

# ---- 打包后自检：关键文件齐全、fused-ssim 未混入、CUDA 源码无非 ASCII 注释 ----
zf = zipfile.ZipFile(DST)
names = zf.namelist()
assert not any("/fused-ssim/" in n for n in names), "fused-ssim 混入了 zip！"
key = [
    "gaussian-splattingWithMask_v2/train_mask.py",
    "gaussian-splattingWithMask_v2/evaluate.py",
    "gaussian-splattingWithMask_v2/colab/ZS601_3DGS_stagesv2.ipynb",
    "gaussian-splattingWithMask_v2/AGENTS.md",
    "gaussian-splattingWithMask_v2/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
    "gaussian-splattingWithMask_v2/submodules/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp",
    "gaussian-splattingWithMask_v2/submodules/simple-knn/setup.py",
]
for k in key:
    assert k in names, "缺失关键文件: " + k
pat = re.compile(r"[\u4e00-\u9fff\uff00-\uffef]")
for n in names:
    if n.startswith("gaussian-splattingWithMask_v2/submodules/diff-gaussian-rasterization/") \
            and n.endswith((".cu", ".h", ".cpp")) and "third_party" not in n:
        assert not pat.search(zf.read(n).decode("utf-8")), "CUDA 源码含非 ASCII 字符: " + n
print("自检通过：fused-ssim 已排除，关键文件齐全，CUDA 源码全 ASCII")
