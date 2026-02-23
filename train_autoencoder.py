# -*- encoding: utf-8 -*-
'''
@Author  :   Hui Li, Jiangnan University
@Contact :   lihui.cv@jiangnan.edu.cn
@File    :   train_autoencoder.py
@Time    :   2024/06/15 16:28:59
'''

# here put the import lib
# Train auto-encoder model for infrarde or visible image

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
from network.net_autoencoder import Auto_Encoder_single

from args_auto import Args as args


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

# 启用 CUDA；AMP 默认为关闭，需对齐 PyTorch 基线时用 FP32。若需 AMP，可将 use_amp 设为 True。
jt.flags.use_cuda = 1 if args.cuda else 0
use_amp = False
jt.flags.amp_level = 3 if use_amp else 0
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

# -------------------------------------------------------
def load_data(path, train_num):
	# train_num for KAIST
	# 根据 type_flag 过滤图像类型：'ir' 只收集 lwir，'vi' 只收集 visible
	imgs_path, _ = utils.list_images_datasets(path, train_num, filter_type=args.type_flag)
	# imgs_path = imgs_path[:train_num]
	random.shuffle(imgs_path)
	return imgs_path


# -------------------------------------------------------
def train(data, img_flag):
	batch_size = args.batch
	step = args.step
	
	# auto-encoder
	model = Auto_Encoder_single(**custom_config_auto)
	if args.resume_model_auto is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_model_auto))
		model.load_parameters(jt.load(args.resume_model_auto))
	
	# 与参考版一致：只优化可训练参数
	trainable_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = jt.optim.Adam(trainable_params, lr=args.lr, weight_decay=0.9)

	print('Start training.....')
	
	# 创建保存路径（每次训练一个独立子文件夹，避免覆盖），并在文件名中加入 pytorch + 时间戳
	training_session_id = time.strftime('%Y%m%d_%H%M%S')
	session_dir_name = f"pytorch_autoencoder_{args.type_flag}_{training_session_id}"
	# 保存到 models/my_trained_models/jittor/autoencoder/<session>
	temp_path_model = os.path.join(args.save_auto_model, "jittor", session_dir_name)
	if os.path.exists(temp_path_model) is False:
		os.makedirs(temp_path_model)

	temp_path_loss = os.path.join(temp_path_model, 'loss')
	if os.path.exists(temp_path_loss) is False:
		os.makedirs(temp_path_loss)
	
	# creating log path
	# 日志存放在 logs/.../jittor/autoencoder
	log_dir = os.path.join(args.save_log_dir, "jittor")
	if os.path.exists(log_dir) is False:
		os.makedirs(log_dir)
	log_file_path = os.path.join(log_dir, 'train_autoencoder_{}_{}.log'.format(args.type_flag, time.strftime('%Y%m%d_%H%M%S')))
	logger = Logger(log_file_path)
	sys.stdout = logger
	print('=' * 80)
	print('Training Log - Autoencoder ({})'.format(args.type_flag))
	print('Start time: {}'.format(time.ctime()))
	print('Log file: {}'.format(log_file_path))
	print('=' * 80)
	
	# 对齐 PyTorch 版本：初始化为 float 0.0
	loss_p4 = 0.
	loss_p5 = 0.
	loss_all = 0.
	
	loss_mat = []
	model.train()
	count = 0
	for e in range(args.epochs):
		lr_cur = utils.adjust_learning_rate(optimizer, e, args.lr)
		img_paths, batch_num = utils.load_dataset(data, batch_size)
		
		for idx in range(batch_num):
			
			image_paths = img_paths[idx * batch_size:(idx * batch_size + batch_size)]
			img = utils.get_train_images(image_paths, height=args.Height, width=args.Width, flag=img_flag)
			
			count += 1
			batch = img
			
			# for DataParallel
			outputs = model.train_module(batch)
			
			img_out = outputs['out']
			recon_loss = outputs['recon_loss']
			ssim_loss = outputs['ssim_loss']
			total_loss = outputs['total_loss']
			loss_mat.append(float(total_loss))
			optimizer.step(total_loss)
			
			# 对齐 PyTorch 版本：直接累积张量（PyTorch会自动转换为float）
			loss_p4 += outputs['recon_loss']
			loss_p5 += outputs['ssim_loss']
			loss_all += total_loss
			
			# # Test
			# if count % 1000 == 0:
			# 	with torch.no_grad():
			# 		test(model_auto_ir, model_auto_vi, model, e + 1)
			# 		print('Done. Testing image data on epoch {}'.format(e + 1))
			
			if count % step == 0:
				# 对齐 PyTorch 版本：先转换为float再除以step，避免Jittor中float+tensor混合类型导致的NaN
				# PyTorch中tensor /= step后仍然是tensor，format时会自动转换；Jittor需要显式转换
				loss_p4 = float(loss_p4) / step
				loss_p5 = float(loss_p5) / step
				loss_all = float(loss_all) / step
				# if e == 0 and count == step:
				# 	viz.line([loss_all.item()], [0.], win='train_loss', opts=dict(title='Total Loss'))
				
				mesg = "{} - Epoch {}/{} - Batch {}/{} - lr:{:.6f} - recon loss: {:.6f} - ssim loss: {:.6f} - total loss: {:.6f} \n". \
					format(time.ctime(), e + 1, args.epochs, idx + 1, batch_num, lr_cur,
				           loss_p4, loss_p5, loss_all)
				
				# viz.line([loss_all.item()], [count], win='train_loss', update='append')
				# img1 = torch.cat((batch[0, :, :, :], img_out[0, :, :, :]), 0)
				# img_or2 = torch.cat((batch_ir[1, :, :, :], batch_vi[1, :, :, :]), 0)
				# img2 = torch.cat((img_or2, img_out[1, :, :, :]), 0)
				# viz.images(img1.view(-1, 1, args.Height, args.Width), win='x')
				
				# ir_sa, vi_sa, ir_ca, vi_ca, c_fe = middle_temp[0], middle_temp[1], middle_temp[2], middle_temp[3], middle_temp[4]
				# img_fe = torch.cat((ir_sa[0, :, :, :], vi_sa[0, :, :, :]), 0)
				# img_fe = torch.cat((img_fe, ir_ca[0, :, :, :]), 0)
				# img_fe = torch.cat((img_fe, vi_ca[0, :, :, :]), 0)
				# img_fe = torch.cat((img_fe, c_fe[0, :, :, :]), 0)
				# viz.images(img_fe.view(-1, 1, args.Height, args.Width), win='y')
				
				# weight = torch.cat((weights[0][0, :, :, :], weights[1][0, :, :, :]), 0)
				# weight_fuse = torch.cat((weight, weights[2][0, :, :, :]), 0)
				# weight_fuse = torch.cat((weight_fuse, max_temp[0, :, :, :]), 0)
				# viz.images(weight_fuse.view(-1, 1, args.Height, args.Width), win='z')
				# viz.images(weights[3][0, :, :, :].view(-1, 1, args.Height, args.Width), win='z1')
				
				print(mesg)
				loss_p4 = 0.
				loss_p5 = 0.
				loss_all = 0.
		
		# with torch.no_grad():
		# 	print('Start. Testing image data on epoch {}'.format(e + 1))
		# 	test(model_auto_ir, model_auto_vi, model, shift_flag, e + 1)
		# 	print('Done. Testing image data on epoch {}'.format(e + 1))
		
		# save loss（文件名中加入 pytorch + 时间戳，以便区分不同训练）
		save_model_filename = f"loss_data_autoencoder_pytorch_e{e}_session_{training_session_id}.mat"
		loss_filename_path = os.path.join(temp_path_loss, save_model_filename)
		scio.savemat(loss_filename_path, {'loss_data': loss_mat})
		# save model
		model.eval()
		# 模型文件名中加入 pytorch + 时间戳，避免覆盖历史模型
		save_model_filename = f"auto_encoder_pytorch_epoch_{e + 1}_{args.type_flag}_{training_session_id}.model"
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
	# True - RGB, False - gray
	if args.channel == 1:
		img_flag = False
	else:
		img_flag = True
	
	path = args.path
	train_num = args.train_num
	data = load_data(path, train_num)
	
	train(data, img_flag)
