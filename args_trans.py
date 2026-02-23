# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : TransFuse
# @File : args_trans.py
# @Time : 2021/11/9 14:15

class Args():
	# For training
	# 使用本地 database 文件夹下的 KAIST 数据集（支持 set00/V000/lwir 等嵌套结构）
	path_ir = ['./database/']  # 递归搜索所有 lwir 子目录，对应的 visible 会自动匹配
	cuda = True
	lr = 0.001
	# 现在进行“非消融”的正式融合训练，将轮数设置为 16 轮
	epochs = 16
	batch = 8  # 调回8，与第一次训练保持一致，便于对比
	# 消融实验开关：
	# use_ablation_cnn: True=使用CNN Fusion（消融版本），False=使用Cross-Attention（原始版本）
	use_ablation_cnn = False  # False=使用标准 Cross-Attention（CAM）
	# use_ablation_no_gra_loss: True=去掉梯度损失，只使用像素损失（梯度损失消融实验）
	# 你现在想“使用所有损失函数”，所以这里必须为 False，开启梯度损失
	use_ablation_no_gra_loss = False
	train_num = 20000
	step = 10
	# Network Parameters
	channel = 1
	Height = 256
	Width = 256
 
	crop_h = 256
	crop_w = 256

	vgg_model_dir = "./models/vgg"
	# 使用你在 Jittor 下重新训练的自编码器模型（IR / VI 各 4 轮）
	# 注意：路径中包含 jittor 子目录和时间戳，以区分不同训练会话
	resume_model_auto_ir = "./models/my_trained_models/autoencoder/jittor/pytorch_autoencoder_ir_20260220_230941/auto_encoder_pytorch_epoch_4_ir_20260220_230941.model"
	resume_model_auto_vi = "./models/my_trained_models/autoencoder/jittor/pytorch_autoencoder_vi_20260221_003703/auto_encoder_pytorch_epoch_4_vi_20260221_003703.model"

	# 从头训练16轮
	resume_model_trans = None
	# 保存你自己训练的融合模型到独立文件夹，避免覆盖原始预训练模型
	# 训练脚本会自动在此路径下创建 transfuse 子文件夹
	save_fusion_model = "./models/my_trained_models"
	save_loss_dir = "./models/my_trained_models/loss"
	# 训练日志保存路径
	save_log_dir = "./logs/my_trained_models/transfuse"
