# -*- encoding: utf-8 -*-
'''
@Author  :   Hui Li, Jiangnan University
@Contact :   lihui.cv@jiangnan.edu.cn
@File    :   args_auto.py
@Time    :   2024/06/15 16:29:31
'''

# here put the import lib

class Args():
	# For training
	# 使用本地 database 文件夹下的 KAIST 数据集（支持 set00/V000/lwir 等嵌套结构）
	path = ['./database/'] # 递归搜索所有 lwir 或 visible 子目录
	# 先训练红外自编码器：'ir'；训练完 IR 后再手动改成 'vi' 训练可见光
	type_flag = 'vi' # 'ir' 训练红外自编码器，'vi' 训练可见光自编码器
	cuda = True
	lr = 0.0001
	epochs = 4
	batch = 2
	step = 10
	w = [1.0, 10000.0, 0.1, 1.0]
	train_num = 40000
	# Network Parameters
	channel = 1
	Height = 256
	Width = 256
	crop_h = 256
	crop_w = 256

	resume_model_auto = None
	# 保存你自己训练的模型到独立文件夹，避免覆盖原始预训练模型
	save_auto_model = "./models/my_trained_models/autoencoder"
	# 训练日志保存路径
	save_log_dir = "./logs/my_trained_models/autoencoder"







