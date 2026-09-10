#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
import weakref
from collections import OrderedDict
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

# 新功能：懒加载全局 LRU 缓存——最多同时保留 _LAZY_CACHE_SIZE 张已解码图片，
# 避免 2694 张图一次性全部常驻内存导致爆 RAM；被淘汰的相机下次访问时从磁盘重新读取
_LAZY_CACHE = OrderedDict()   # id(camera) -> weakref(camera)
_LAZY_CACHE_SIZE = 100

def set_lazy_cache_size(n):
    """设置懒加载缓存容量（train_mask.py 启动时根据 --lazy_cache 调用）"""
    global _LAZY_CACHE_SIZE
    _LAZY_CACHE_SIZE = max(1, int(n))

_MASK_RESCUE_WARNED = False

def _rescue_mask(mask):
    """新功能：mask 编码抢救——若 mask 最大值远低于 0.5，判定为 0/1 值 mask 被按 0/255 读入
    （前景像素仅 1/255≈0.004），用极小阈值重新二值化；
    正常的 0/255 mask（最大值=1）与全零 mask（保持全零）不受影响"""
    global _MASK_RESCUE_WARNED
    if mask.numel() > 0 and float(mask.max()) < 0.5:
        mask = (mask > (0.5 / 255.0)).float()
        if not _MASK_RESCUE_WARNED:
            print("[warning] 检测到低值 mask（疑似 0/1 编码被当作 0/255 读入），已自动重新二值化；"
                  "后续同类 mask 不再提示")
            _MASK_RESCUE_WARNED = True
    return mask

def _load_mask(pil_mask, resolution):
    """统一的 mask 加载入口：抢救编码后取反。
    本数据集 mask 语义：黑(0)=有效像素（静态主体，需保留），白(255)=动态物体（人，需剔除），
    与原版仓库“白=保留”的假设相反；取反后 mask=1 表示有效像素，gt=image*mask 即
    “有效像素以原图 RGB 参与 loss，无效区域置黑”，训练/val/test 口径一致"""
    return 1.0 - _rescue_mask(PILtoTorch(pil_mask, resolution))

def _lazy_evict():
    """淘汰最久未使用的缓存项，并清空对应相机的图片引用以释放内存"""
    while len(_LAZY_CACHE) > _LAZY_CACHE_SIZE:
        _, ref = _LAZY_CACHE.popitem(last=False)
        cam = ref()
        if cam is not None:
            cam._gt_image = None
            cam._alpha_mask = None

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params,  image, alpha_mask,
                 invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp=False, is_test_dataset=False, is_test_view=False,
                 lazy=False, image_path="", mask_path="",
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # 新功能：懒加载模式——初始化时不解码图片，访问 original_image/alpha_mask 时才从磁盘读取，
        # 已解码图片放入全局 LRU 缓存（容量默认 100 张），内存占用可控
        self.lazy = lazy
        self._gt_image = None
        self._alpha_mask = None
        if lazy:
            self._image_path = image_path
            self._mask_path = mask_path
            self._lazy_resolution = resolution
            self.image_width = resolution[0]
            self.image_height = resolution[1]
            self.invdepthmap = None
            self.depth_reliable = False
        else:
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            if alpha_mask is not None:
                self.alpha_mask = _load_mask(alpha_mask, resolution).to(self.data_device)
            elif resized_image_rgb.shape[0] == 4:
                self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            else:
                self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...]).to(self.data_device)

            if train_test_exp and is_test_view:
                if is_test_dataset:
                    self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
                else:
                    self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

            self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if self.alpha_mask is not None:
                self.original_image *= self.alpha_mask

            self.invdepthmap = None
            self.depth_reliable = False
            if invdepthmap is not None and depth_params is not None and depth_params["scale"] > 0:
                invdepthmapScaled = invdepthmap * depth_params["scale"] + depth_params["offset"]
                invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
                invdepthmapScaled[invdepthmapScaled < 0] = 0
                if invdepthmapScaled.ndim != 2:
                    invdepthmapScaled = invdepthmapScaled[..., 0]
                self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)

                if self.alpha_mask is not None:
                    self.depth_mask = self.alpha_mask.clone()
                else:
                    self.depth_mask = torch.ones_like(self.invdepthmap > 0)

                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_mask *= 0
                else:
                    self.depth_reliable = True

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(self.data_device)
        self.camera_center = self.world_view_transform.inverse()[3, :3].to(self.data_device)

    # 属性封装：懒加载模式下访问时才触发磁盘读取，外部调用方（train/evaluate）无需修改
    @property
    def original_image(self):
        if self.lazy and self._gt_image is None:
            self._load_lazy()
        return self._gt_image

    @original_image.setter
    def original_image(self, value):
        self._gt_image = value

    @property
    def alpha_mask(self):
        if self.lazy and self._alpha_mask is None:
            self._load_lazy()
        return self._alpha_mask

    @alpha_mask.setter
    def alpha_mask(self, value):
        self._alpha_mask = value

    def _load_lazy(self):
        """懒加载：从磁盘读取图片/mask 并解码，乘上 mask 后写入全局 LRU 缓存"""
        key = id(self)
        ref = _LAZY_CACHE.get(key)
        if ref is not None and ref() is self:
            _LAZY_CACHE.move_to_end(key)
            return
        image = Image.open(self._image_path)
        resized = PILtoTorch(image, self._lazy_resolution)
        gt = resized[:3, ...].clamp(0.0, 1.0).to(self.data_device)
        if self._mask_path != "":
            mask = _load_mask(Image.open(self._mask_path), self._lazy_resolution).to(self.data_device)
        else:
            mask = torch.ones_like(resized[0:1, ...]).to(self.data_device)
        gt = gt * mask
        self._gt_image = gt
        self._alpha_mask = mask
        _LAZY_CACHE[key] = weakref.ref(self)
        _LAZY_CACHE.move_to_end(key)
        _lazy_evict()
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

