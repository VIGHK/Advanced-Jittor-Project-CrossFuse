# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : FuseTrans
# @File : train.py
# @Time : 2021/5/14 14:42

# Train fusion models (CAM and decoder)

import os
import scipy.io as scio
import jittor as jt
from jittor import nn
import time
import sys
# pytohn -m visdom.server
# from visdom import Visdom

from tools import utils
import random
from network.net_conv_trans import Trans_FuseNet, Trans_FuseNet_CNN
from network.net_autoencoder import Auto_Encoder_single
from network.loss import Gradient_loss, Order_loss, Patch_loss

from args_trans import Args as args


# -------------------------------------------------------
# 日志保存类：同时输出到控制台和文件
class Logger(object):
	def __init__(self, log_file_path):
		self.terminal = sys.stdout
		self.log_file = open(log_file_path, 'a', encoding='utf-8')
	
	def write(self, message):
		self.terminal.write(message)
		self.log_file.write(message)
		self.log_file.flush()
	
	def flush(self):
		self.terminal.flush()
		self.log_file.flush()
	
	def close(self):
		self.log_file.close()

# -------------------------------------------------------
# Auto-Encoder
custom_config_auto = {
	"in_channels": 1,
	"out_channels": 1,
	"en_out_channels1": 32,
	"en_out_channels": 64,
	"num_layers": 3,
	"dense_out": 128,
	"part_out": 128,
	"train_flag": True,
}
# Trans module
custom_config = {
	"en_out_channels1": 32,
	"out_channels": 1,
	"part_out": 128,
	"train_flag": True,
	
	"img_size": 32,
	"patch_size": 2,
	"depth_self": 1,
	"depth_cross": 1,
	"n_heads": 16,
	"qkv_bias": True,
	"mlp_ratio": 4,
	"p": 0.,
	"attn_p": 0.,
	"use_ablation_no_gra_loss": args.use_ablation_no_gra_loss,  # 消融实验：是否去掉梯度损失
}


# -------------------------------------------------------
def load_data(path, train_num):
	# train_num for KAIST
	# 对齐 PyTorch 版本：不使用 filter_type，直接加载所有图像
	imgs_path, _ = utils.list_images_datasets(path, train_num)
	# imgs_path = imgs_path[:train_num]
	random.shuffle(imgs_path)
	return imgs_path


# -------------------------------------------------------
def test(model_auto_ir, model_auto_vi, model, shift_flag, e):
	img_flag = False
	test_path_ir = './images/21_pairs_tno/ir'
	test_path_vi = './images/21_pairs_tno/vis'
	ir_pathes, ir_names = utils.list_images_test(test_path_ir)
	# ---------------------------------------------------
	output_path1 = './output/transfuse'
	if os.path.exists(output_path1) is False:
		os.mkdir(output_path1)
	output_path = output_path1 + '/training_21_tno_epoch_' + str(e)
	if os.path.exists(output_path) is False:
		os.mkdir(output_path)
	
	for ir_name in ir_names:
		vi_name = ir_name.replace('IR', 'VIS')
		ir_path = os.path.join(test_path_ir, ir_name)
		vi_path = os.path.join(test_path_vi, vi_name)
		# for training phase
		ir_img = utils.get_train_images(ir_path, height=args.Height, width=args.Width, flag=img_flag)
		vi_img = utils.get_train_images(vi_path, height=args.Height, width=args.Width, flag=img_flag)
		# Jittor: 不需要 Variable 和 .cuda()，张量自动在 GPU 上
		
		# ---------------------------------------------
		ir_sh, ir_de = model_auto_ir(ir_img)
		vi_sh, vi_de = model_auto_vi(vi_img)
		outputs = model(ir_de, ir_sh, vi_de, vi_sh, shift_flag)
		out = outputs['out']
		# ---------------------------------------------
		path_out_ir = output_path + '/results_transfuse_' + ir_name
		utils.save_image(out, path_out_ir)


# -------------------------------------------------------
def train(data, img_flag):
	batch_size = args.batch
	step = args.step
	
	# 根据消融实验开关选择模型
	if args.use_ablation_cnn:
		print('=' * 80)
		print('ABLATION EXPERIMENT: Using CNN Fusion instead of Cross-Attention Module')
		if args.use_ablation_no_gra_loss:
			print('ABLATION: Gradient Loss (L_gra) is DISABLED')
		print('=' * 80)
		model = Trans_FuseNet_CNN(**custom_config)
	else:
		print('=' * 80)
		print('STANDARD TRAINING: Using Cross-Attention Module (CAM)')
		if args.use_ablation_no_gra_loss:
			print('ABLATION: Gradient Loss (L_gra) is DISABLED')
		print('=' * 80)
		model = Trans_FuseNet(**custom_config)
	# model = torch.nn.DataParallel(model_or, list(range(torch.cuda.device_count()))).cuda()
	shift_flag = False
	if args.resume_model_trans is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_model_trans))
		model.load_parameters(jt.load(args.resume_model_trans))
	# auto-encoder
	model_auto_ir = Auto_Encoder_single(**custom_config_auto)
	model_auto_vi = Auto_Encoder_single(**custom_config_auto)
	# model_auto_ir = torch.nn.DataParallel(model_auto_ir_or, list(range(torch.cuda.device_count()))).cuda()
	# model_auto_vi = torch.nn.DataParallel(model_auto_vi_or, list(range(torch.cuda.device_count()))).cuda()
	
	if args.resume_model_auto_ir is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_model_auto_ir))
		model_auto_ir.load_parameters(jt.load(args.resume_model_auto_ir))
		model_auto_vi.load_parameters(jt.load(args.resume_model_auto_vi))
		# 打印一条权重均值用来确认权重确实加载（记录在日志前，仍会输出到控制台）
		# 对齐 PyTorch 版本：移除参数均值打印（PyTorch版本没有此功能）
		# 如果需要验证权重加载，可以在日志中手动检查
	
	# ------------------------------------------------------
	# 对齐 PyTorch 版本：PyTorch 使用 filter(lambda x: x.requires_grad, model.parameters())
	# 但在 PyTorch 中，conv_gra 等固定卷积核的 requires_grad 可能仍为 True（虽然梯度为0）
	# Jittor 中，stop_grad() 的参数可能 requires_grad=False，导致被过滤
	# 为了对齐行为，我们直接使用所有参数，让 Jittor 优化器自动处理 stop_grad 的参数
	# 注意：PyTorch 版本也使用 weight_decay=0.9，所以保持原值
	# 为完全对齐 PyTorch：使用全部参数，交由优化器处理 stop_grad（与 PyTorch 参数列表一致）
	trainable_params = list(model.parameters())
	optimizer = jt.optim.Adam(trainable_params, lr=args.lr, weight_decay=0.9)
	# ------------------------------------------------------
	gra_loss = Gradient_loss(custom_config['out_channels'])
	order_loss = Order_loss(custom_config['out_channels'])
	
	# visdom
	# viz = Visdom()
	
	model_auto_ir.eval()
	model_auto_vi.eval()
	print('Start training.....')
	
	# creating save path（外层根目录，增加 jittor 子目录）
	temp_path_model1 = os.path.join(args.save_fusion_model, "jittor")
	if os.path.exists(temp_path_model1) is False:
		os.makedirs(temp_path_model1)
	# 生成训练会话ID（时间戳），用于区分不同的训练会话，避免覆盖之前的模型
	training_session_id = time.strftime('%Y%m%d_%H%M%S')
	# 消融实验：使用独立的文件夹结构，并在文件夹名中加入 pytorch 标记
	if args.use_ablation_cnn:
		# 消融实验模型保存在独立的 ablation 文件夹下
		temp_path_model = os.path.join(temp_path_model1, 'ablation', f"pytorch_transfuse_cnn_{training_session_id}")
	elif args.use_ablation_no_gra_loss:
		# 梯度损失消融实验：使用独立的文件夹
		temp_path_model = os.path.join(temp_path_model1, 'ablation', f"pytorch_no_gra_loss_{training_session_id}")
	else:
		# 标准训练模型保存在 transfuse 文件夹下
		temp_path_model = os.path.join(temp_path_model1, 'transfuse', f"pytorch_transfuse_{training_session_id}")
	# 每次训练创建一个独立的文件夹，存放该次训练的所有模型和loss文件
	if os.path.exists(temp_path_model) is False:
		os.makedirs(temp_path_model)
	
	temp_path_loss = os.path.join(temp_path_model, 'loss')
	if os.path.exists(temp_path_loss) is False:
		os.makedirs(temp_path_loss)
	
	# creating log path
	if args.use_ablation_cnn:
		# 消融实验日志保存在独立的文件夹
		log_dir = os.path.join('./logs/my_trained_models/jittor/ablation/transfuse_cnn')
	elif args.use_ablation_no_gra_loss:
		# 梯度损失消融实验日志保存在独立的文件夹
		log_dir = os.path.join('./logs/my_trained_models/jittor/ablation/no_gra_loss')
	else:
		log_dir = os.path.join(args.save_log_dir, "jittor")
	if os.path.exists(log_dir) is False:
		os.makedirs(log_dir)
	log_file_path = os.path.join(log_dir, 'train_transfuse_{}.log'.format(training_session_id))
	logger = Logger(log_file_path)
	sys.stdout = logger
	print('=' * 80)
	print('Training Log - Fusion Network (TransFuse)')
	if args.use_ablation_no_gra_loss:
		print('ABLATION EXPERIMENT: Training WITHOUT Gradient Loss (L_gra)')
	print('Start time: {}'.format(time.ctime()))
	print('Log file: {}'.format(log_file_path))
	print('Training session ID: {}'.format(training_session_id))
	print('Model save directory: {}'.format(temp_path_model))
	print('Loss save directory: {}'.format(temp_path_loss))
	print('=' * 80)
	
	loss_p4 = 0.
	loss_p5 = 0.
	loss_p6 = 0.
	loss_p7 = 0.
	loss_p8 = 0.
	loss_p9 = 0.
	loss_p10 = 0.
	loss_all = 0.
	
	loss_mat = []
	model.train()
	count = 0
	for e in range(args.epochs):
		lr_cur = utils.adjust_learning_rate(optimizer, e, args.lr)
		img_paths, batch_num = utils.load_dataset(data, batch_size)
		
		for idx in range(batch_num):
			
			image_paths_ir = img_paths[idx * batch_size:(idx * batch_size + batch_size)]
			img_ir = utils.get_train_images(image_paths_ir, height=args.Height, width=args.Width, flag=img_flag)
			
			image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]
			img_vi = utils.get_train_images(image_paths_vi, height=args.Height, width=args.Width, flag=img_flag)
			
			count += 1
			batch_ir = img_ir
			batch_vi = img_vi
			
			# 保持与原版一致：自编码器前向不参与梯度
			with jt.no_grad():
				ir_sh, ir_de = model_auto_ir(batch_ir)
				vi_sh, vi_de = model_auto_vi(batch_vi)

			# for DataParallel
			outputs = model.train_module(batch_ir, batch_vi, ir_sh, vi_sh, ir_de, vi_de, shift_flag, gra_loss, order_loss)
			
			img_out = outputs['out']
			weights = outputs['weight']
			middle_temp = outputs['middle_temp']
			total_loss = outputs['total_loss']
			
			# -------- 调试：仅前 3 个 batch 打印输出/输入范围及损失，便于比对 PyTorch --------
			if count <= 3:
				out_min = float(jt.min(img_out))
				out_max = float(jt.max(img_out))
				ir_min = float(jt.min(batch_ir))
				ir_max = float(jt.max(batch_ir))
				vi_min = float(jt.min(batch_vi))
				vi_max = float(jt.max(batch_vi))
				print(f"[DEBUG] batch {count} out min/max: {out_min:.3f}/{out_max:.3f} | "
				      f"ir min/max: {ir_min:.3f}/{ir_max:.3f} | vi min/max: {vi_min:.3f}/{vi_max:.3f}")
				print(f"[DEBUG] batch {count} pix_loss: {float(outputs['pix_loss']):.3f}, "
				      f"gra_loss: {float(outputs['gra_loss']):.3f}, total_loss: {float(total_loss):.3f}")
			# ---------------------------------------------------------------------------
			
			loss_mat.append(float(total_loss))
			optimizer.step(total_loss)
			
			# 对齐 PyTorch 版本：直接累积张量（PyTorch会自动转换为float）
			loss_p4 += outputs['pix_loss']
			loss_p5 += outputs['sh_loss']
			loss_p6 += outputs['mi_loss']
			loss_p7 += outputs['de_loss']
			loss_p8 += outputs['fea_loss']
			loss_p9 += outputs['gra_loss']
			loss_p10 += outputs['mean_loss']
			loss_all += total_loss
			
			# # Test
			# if count % 1000 == 0:
			# 	with torch.no_grad():
			# 		test(model_auto_ir, model_auto_vi, model, e + 1)
			# 		print('Done. Testing image data on epoch {}'.format(e + 1))
			
			if count % step == 0:
				loss_p4 /= step
				loss_p5 /= step
				loss_p6 /= step
				loss_p7 /= step
				loss_p8 /= step
				loss_p9 /= step
				loss_p10 /= step
				loss_all /= step
				# if e == 0 and count == step:
				# 	viz.line([loss_all.item()], [0.], win='train_loss', opts=dict(title='Total Loss'))
				
				mesg = "{} - Epoch {}/{} - Batch {}/{} - lr:{:.6f} - pix loss: {:.6f} - gra loss: {:.6f} - mean loss:{:.6f}" \
				       " - shallow loss: {:.6f} - middle loss: {:.6f}\n" \
				       "deep loss: {:.6f} - fea loss: {:.6f} \t total loss: {:.6f} \n". \
					format(time.ctime(), e + 1, args.epochs, idx + 1, batch_num, lr_cur,
				           loss_p4, loss_p9, loss_p10, loss_p5, loss_p6, loss_p7, loss_p8, loss_all)
				
				# viz.line([loss_all.item()], [count], win='train_loss', update='append')
				img_or1 = jt.concat((batch_ir[0, :, :, :], batch_vi[0, :, :, :]), 0)
				img1 = jt.concat((img_or1, img_out[0, :, :, :]), 0)
				# img_or2 = torch.cat((batch_ir[1, :, :, :], batch_vi[1, :, :, :]), 0)
				# img2 = torch.cat((img_or2, img_out[1, :, :, :]), 0)
				# viz.images(img1.view(-1, 1, args.Height, args.Width), win='x')
				
				# 训练阶段不做中间特征可视化，避免无关的 reshape/concat 异常
				# ir_sa, vi_sa, ir_ca, vi_ca, c_fe = middle_temp[0], middle_temp[1], middle_temp[2], middle_temp[3], middle_temp[4]
				# img_fe = jt.concat((ir_sa[0, :, :, :], vi_sa[0, :, :, :]), 0)
				# img_fe = jt.concat((img_fe, ir_ca[0, :, :, :]), 0)
				# img_fe = jt.concat((img_fe, vi_ca[0, :, :, :]), 0)
				# img_fe = jt.concat((img_fe, c_fe[0, :, :, :]), 0)
				# weight = jt.concat((weights[0][0, :, :, :], weights[1][0, :, :, :]), 0)
				# weight_fuse = jt.concat((weight, weights[2][0, :, :, :]), 0)
				
				print(mesg)
				loss_p4 = 0.
				loss_p5 = 0.
				loss_p6 = 0.
				loss_p7 = 0.
				loss_p8 = 0.
				loss_p9 = 0.
				loss_p10 = 0.
				loss_all = 0.
		
		# with torch.no_grad():
		# 	print('Start. Testing image data on epoch {}'.format(e + 1))
		# 	test(model_auto_ir, model_auto_vi, model, shift_flag, e + 1)
		# 	print('Done. Testing image data on epoch {}'.format(e + 1))
		
		# save loss：在文件名中加入 pytorch + 时间戳，便于区分不同训练
		save_model_filename = f"loss_data_trans_pytorch_e{e}_session_{training_session_id}.mat"
		loss_filename_path = os.path.join(temp_path_loss, save_model_filename)
		scio.savemat(loss_filename_path, {'loss_data': loss_mat})
		# save model：文件名中同时包含 pytorch + epoch + 时间戳，完全避免覆盖
		model.eval()
		# Jittor: 不需要 .cpu()，直接保存即可
		save_model_filename = f"fusetrans_pytorch_epoch_{e + 1}_{training_session_id}.model"
		save_model_path = os.path.join(temp_path_model, save_model_filename)
		jt.save(model.state_dict(), save_model_path)
		##############
		model.train()
		print("\nCheckpoint, trained model saved at: " + save_model_path)
	
	print("\nDone, TransFuse training phase.")
	# 恢复标准输出并关闭日志文件
	sys.stdout = logger.terminal
	logger.close()
	print("Training completed. Log saved to: {}".format(log_file_path))


if __name__ == "__main__":
	# 启用 CUDA/AMP（Jittor）- 必须在导入args之后设置
	jt.flags.use_cuda = 1 if args.cuda else 0
	# AMP 开关：开启 AMP 加速训练，节省显存并提升速度
	# amp_level=3 表示使用 O3 优化级别（最高性能）
	use_amp = True
	jt.flags.amp_level = 3 if use_amp else 0
	
	# True - RGB, False - gray
	if args.channel == 1:
		img_flag = False
	else:
		img_flag = True
	
	path = args.path_ir
	train_num = args.train_num
	data = load_data(path, train_num)
	
	train(data, img_flag)
