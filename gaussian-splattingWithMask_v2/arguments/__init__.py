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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        #这里是加了mask部分
        self._alpha_masks = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        # 新功能 1：2D 椭球初始化（默认关闭，不开启时行为与原版完全一致）
        self.init_2d = False
        # z 轴缩放初始值（log 空间），exp 后约 4.5e-5，参考 2DGS
        self.init_2d_z = -10.0
        # init_2d 专用初始不透明度：激光点云初始化已可靠，从 0.5 起步
        # （非 init_2d 时仍用 3DGS 默认 0.1，不受此参数影响）
        self.init_2d_opacity = 0.5
        # 新功能：2D 椭球形状约束（默认关闭）——训练中每步把 z 轴缩放复位为极小值，
        # 保持高斯始终扁平；与 --init_2d 独立：B 组两个都开，C 组只开 --init_2d（允许厚度自由优化）
        self.freeze_2d_z = False
        # 新功能 2：图片懒加载（默认关闭）——按需从磁盘读取图片并 LRU 缓存，
        # 避免大量图片一次性全部常驻内存导致爆 RAM（适合 Colab 等内存受限环境）
        self.lazy_load = False
        # 懒加载模式下最多缓存的已解码图片张数（约 100 张 * 2.5MB ≈ 0.3GB，加 mask 约 1GB）
        self.lazy_cache = 100
        # 阶段2新功能：法向量约束 loss 权重（参考 2DGS）：
        # --lambda_scale：z 轴厚度正则 |exp(scaling)[:,2]| 均值的权重；
        # --lambda_normal：渲染法向与深度导出法向一致性 loss 的权重；两者为 0 时对应项跳过。
        # --normal_start_iter：法向 loss 延迟启用轮次（对齐 2DGS 原版：前期深度不可靠，
        # 深度导出法向是噪声，过早启用会扰乱旋转优化；冒烟验证时可传 0 强制启用）。
        # 注意：若重跑 A/B 组 baseline 需显式传 --lambda_scale 0 --lambda_normal 0
        self.lambda_scale = 0.1
        self.lambda_normal = 0.1
        self.normal_start_iter = 7000
        # 阶段2新增：形状约束（防大椭球）——逐高斯惩罚最长尺度轴 exp 后均值的权重；
        # 默认 0 关闭（不显式开启时不影响既有 A/B/C/D 组口径）；建议从 0.01 起调
        self.lambda_size = 0.0
        # 阶段2新增（高斯级形态正则，目标：小扁盘贴表面 + 邻居法向趋同）：
        # --lambda_flat：每高斯最薄轴尺度均值正则（强制每个高斯至少有一个轴薄，
        # 不依赖轴序号，比只压第 3 轴的 lambda_scale 更鲁棒）；建议 0.05 起调
        # --lambda_smooth：kNN 邻居法向趋同正则（同一表面相邻高斯法向趋同，mean(1-|dot|)）；建议 0.05 起调
        # --smooth_knn_k / --smooth_every：邻居数 / 邻居图重建间隔（致密化后自动重建）
        self.lambda_flat = 0.0
        self.lambda_smooth = 0.0
        self.smooth_knn_k = 8
        self.smooth_every = 500
        # 阶段2新增：锚点法向 loss——把 anchor_iter 轮（默认 1，即初始化状态）的每视图渲染法向图
        # 固定为 GT 靶标，约束当前渲染法向向其靠拢（mean(1-cos)）。用于打断
        # “深度-法向一致性两侧靶标皆自生成”的自增强恶化回路；激光点云初始法向可信，建议 0.2 起调
        self.lambda_anchor = 0.0
        self.anchor_iter = 1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
