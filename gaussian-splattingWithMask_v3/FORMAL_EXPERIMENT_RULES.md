# 正式实验规则

本规则适用于v3-d分支的15万步正式实验。

1. 数据先从Google Drive复制到Colab本地。训练、验证和test必须使用显式且互斥的相机列表，并在run_manifest.json记录内容身份。
2. 固定val在初始化、每5000步和最终步输出10个相机的RGB、1σ彩色椭球、深度和法向量PNG。默认不保存geometry.npz。
3. checkpoint和Gaussian PLY每50000步保存，即50000、100000、150000。不得覆盖已有输出目录。
4. 150000步完成后，在最终模型上评估完整test列表。保存逐相机masked PSNR、masked MAE、zero-mask full-image SSIM，以及MEAN、MEDIAN、MIN、MAX汇总。
5. 最差10个相机按masked PSNR升序选择，同分按image_name排序。只为这10个相机保存RGB、1σ彩色椭球、深度和法向量；test不保存NPZ。
6. experiment_summary.md必须写明相对baseline的功能和超参数变化、相对原始默认值的关键变化、最终test结果、最差10相机、val走势和几何/增密日志分析。
7. baseline数值差只允许在输入内容身份、test相机有序名单和指标协议完全一致时计算；不一致时必须拒绝比较。
8. completed.json只有在最终test和总结文档成功写入后才可设置final_test_complete=true。之后执行os.sync，确认Drive产物存在，再释放Colab GPU。
9. 200步冒烟显式设置--final_test off，但仍要验证CUDA扩展、D开关、val四类图、损失有限、scale bounds、受控增密状态可保存。
