# Colab 工作流

1. 建立唯一会话和 `RUN_ID`。
2. 打印环境并执行 `scripts/check_environment.py`。
3. 挂载 Drive；认证和授权由用户本人完成。
4. 克隆固定 Git commit，不使用浮动分支作为正式运行身份。
5. 安装依赖，详细编译 CUDA 扩展，运行测试。
6. 将数据从 Drive 复制到 `/content/datasets/<scene>`。
7. 在本地盘执行预处理与训练。
8. 先做 200 步冒烟；正式训练使用另一个新目录。
9. 执行完整 test、汇总和 `verify_run_outputs.py`。
10. 复制结果到 Drive 新目录并回读关键文件。
11. 只有回读成功后才停止会话。

Notebook 必须使用显式单元格，标题写明环境、数据、参数和阶段；不能把主要工作隐藏在一个没有输出的后台调用中。
