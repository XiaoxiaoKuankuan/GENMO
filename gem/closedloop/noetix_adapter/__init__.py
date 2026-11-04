"""接入 noetix_supermimic 的独立音乐参考 Adapter，不导入旧版 Goal570 桥接代码。

本包使用用户指定的 40k Actor 及其训练源码、机器人和世界模型；Actor 始终冻结。
数据准备保留 GENMO 名义参考来源，训练仅更新新 Adapter 和独立世界模型。
"""
