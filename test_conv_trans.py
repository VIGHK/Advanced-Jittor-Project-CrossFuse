# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project: CrossFuse
# @File: test_conv_trans
# @Time: 2023/2/28 18:39

import os
import re
import jittor as jt
import numpy as np
from network.net_autoencoder import Auto_Encoder_single
from network.net_conv_trans import Trans_FuseNet, Trans_FuseNet_CNN
from tools import utils
from args_trans import Args as args

k = 10


# imagenet_labels = dict(enumerate(open("classes.txt")))


def load_model(custom_config_auto, custom_config_trans, model_path_auto_ir, model_path_auto_vi, model_path_trans, use_cnn_fusion=False):
    model_auto_ir = Auto_Encoder_single(**custom_config_auto)
    model_auto_ir.load_parameters(jt.load(model_path_auto_ir))
    model_auto_ir.eval()
    
    model_auto_vi = Auto_Encoder_single(**custom_config_auto)
    model_auto_vi.load_parameters(jt.load(model_path_auto_vi))
    model_auto_vi.eval()
    # ---------------------------------------------------------
    # 根据模型类型选择对应的网络结构
    if use_cnn_fusion:
        model_trans = Trans_FuseNet_CNN(**custom_config_trans)
    else:
        model_trans = Trans_FuseNet(**custom_config_trans)
    model_trans.load_parameters(jt.load(model_path_trans))
    model_trans.eval()
    return model_auto_ir, model_auto_vi, model_trans


def test(model_auto_ir, model_auto_vi, model_trans, shift_flag, ir_path, vi_path, ir_name, output_path, img_flag):

    ir_img = utils.get_train_images(ir_path, None, None)
    if img_flag:
        vi_img, vi_cb, vi_cr = utils.get_test_images_color(vi_path, None, None)
    else:
        vi_img = utils.get_train_images(vi_path, None, None)
    
    # ---------------------------------------------
    ir_sh, ir_de = model_auto_ir(ir_img)
    vi_sh, vi_de = model_auto_vi(vi_img)
    outputs = model_trans(ir_de, ir_sh, vi_de, vi_sh, shift_flag)
    img_out = outputs['out']
    # # ---------------------------------------------
    
    # ---------------------------------------------
    path_out = output_path + '/results_transfuse_'
    if img_flag:
        utils.save_image_color(img_out, vi_cb, vi_cr, path_out + ir_name)
    else:
        utils.save_image(img_out, path_out + ir_name)



if __name__ == "__main__":
    jt.flags.use_cuda = 1 if args.cuda else 0
    # Auto-Encoder
    custom_config_auto = {
        "in_channels": 1,
        "out_channels": 1,
        "en_out_channels1": 32,
        "en_out_channels": 64,
        "num_layers": 3,
        "dense_out": 128,
        "part_out": 128,
        "train_flag": False,
    }
    # Trans module
    custom_config_trans = {
        "en_out_channels1": 32,
        "out_channels": 1,
        "part_out": 128,
        "train_flag": False,
        
        "img_size": 32,
        "patch_size": 2,
        "depth_self": 1,
        "depth_cross": 1,
        "n_heads": 16,
        "qkv_bias": True,
        "mlp_ratio": 4,
        "p": 0.,
        "attn_p": 0.,
        "use_ablation_no_gra_loss": False,  # 测试时不需要，仅用于兼容模型初始化
    }
    
    # 使用 Jittor 训练的自编码器模型
    resume_model_auto_ir = "./models/my_trained_models/autoencoder/jittor/pytorch_autoencoder_ir_20260219_164733/auto_encoder_pytorch_epoch_4_ir_20260219_164733.model"
    resume_model_auto_vi = "./models/my_trained_models/autoencoder/jittor/pytorch_autoencoder_vi_20260219_181054/auto_encoder_pytorch_epoch_4_vi_20260219_181054.model"
    
    data_type = ['21_pairs_tno', '40_vot_tno', 'M3FD_Fusion']
    d_type = data_type[2]  # 测试 'M3FD_Fusion'（彩色图像），索引0是'21_pairs_tno'，索引1是'40_vot_tno'（灰度）
    # 彩色图像测试需要设置 img_flag = True
    if d_type.__contains__('M3FD_Fusion'):
        img_flag = True
    else:
        img_flag = False
    
    test_path_ir = './images/' + d_type + '/ir'
    test_path_vi = './images/' + d_type + '/vis'
    ir_pathes, ir_names = utils.list_images_test(test_path_ir)
    
    # ============================================================
    # 三种测试情况（根据需要修改模型路径）：
    # 1. 标准训练：Cross-Attention + 完整损失 (L_pix + L_gra)
    # 2. 梯度损失消融：Cross-Attention + 仅像素损失 (L_pix)
    # 3. CNN Fusion消融：CNN Fusion + 完整损失 (L_pix + L_gra)
    # ============================================================
    
    # 情况1：标准训练模型
    # 使用您训练的模型：pytorch_transfuse_20260220_021031
    model_path_trans = "./models/my_trained_models/jittor/transfuse/pytorch_transfuse_20260220_021031/fusetrans_pytorch_epoch_16_20260220_021031.model"
    
    # 情况2：梯度损失消融模型（去掉gra loss）
    # model_path_trans = "./models/my_trained_models/ablation/no_gra_loss/20251221_033618/fusetrans_epoch_8.model"
    
    # 情况3：CNN Fusion消融模型（替换Cross-Attention）
    # model_path_trans = "./models/my_trained_models/ablation/transfuse_cnn/20251221_001106/fusetrans_epoch_8.model"
    
    # 如果需要使用原始预训练模型，可以改为：
    # model_path_trans = "./models/transfuse/fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model"
    
    # ----------------------------------------------------
    # 从模型路径中自动识别模型类型，并生成对应的输出文件夹名称
    # 格式说明：
    # - 标准训练：{dataset}_epoch{epoch}_{session_id}
    # - 梯度损失消融：{dataset}_no_gra_loss_epoch{epoch}_{session_id}
    # - CNN Fusion消融：{dataset}_cnn_fusion_epoch{epoch}_{session_id}
    
    is_no_gra_loss = 'ablation/no_gra_loss' in model_path_trans
    is_cnn_fusion = 'ablation/transfuse_cnn' in model_path_trans
    use_cnn_fusion_model = is_cnn_fusion  # 用于加载模型
    
    # 匹配路径格式：支持多种格式
    # 格式1: .../pytorch_transfuse_20260220_021031/fusetrans_pytorch_epoch_16_20260220_021031.model
    # 格式2: .../20251220_140533/fusetrans_epoch_16.model
    match1 = re.search(r'pytorch_transfuse_(\d{8}_\d{6})/fusetrans_pytorch_epoch_(\d+)_\d{8}_\d{6}', model_path_trans)
    match2 = re.search(r'(\d{8}_\d{6})/fusetrans_epoch_(\d+)', model_path_trans)
    
    if match1:
        session_id = match1.group(1)  # 20260220_021031
        epoch_num = match1.group(2)   # 16
        if is_no_gra_loss:
            data_type_file = '/' + d_type + '_no_gra_loss_epoch' + epoch_num + '_' + session_id
        elif is_cnn_fusion:
            data_type_file = '/' + d_type + '_cnn_fusion_epoch' + epoch_num + '_' + session_id
        else:
            data_type_file = '/' + d_type + '_epoch' + epoch_num + '_' + session_id
    elif match2:
        session_id = match2.group(1)  # 20251220_140533
        epoch_num = match2.group(2)   # 16 或 8
        if is_no_gra_loss:
            data_type_file = '/' + d_type + '_no_gra_loss_epoch' + epoch_num + '_' + session_id
        elif is_cnn_fusion:
            data_type_file = '/' + d_type + '_cnn_fusion_epoch' + epoch_num + '_' + session_id
        else:
            data_type_file = '/' + d_type + '_epoch' + epoch_num + '_' + session_id
    else:
        # 如果路径格式不匹配，使用默认格式
        if is_no_gra_loss:
            data_type_file = '/' + d_type + '_no_gra_loss_my_trained_transfuse'
        elif is_cnn_fusion:
            data_type_file = '/' + d_type + '_cnn_fusion_my_trained_transfuse'
        else:
            data_type_file = '/'+ d_type + '_my_trained_transfuse'
    
    # ---------------------------------------------------
    # 为 Jittor 版本设置专门的输出文件夹
    output_path1 = './output/crossfuse_test/jittor'
    if os.path.exists(output_path1) is False:
        os.makedirs(output_path1, exist_ok=True)
    output_path = output_path1 + data_type_file
    if os.path.exists(output_path) is False:
        os.makedirs(output_path, exist_ok=True)
    # ---------------------------------------------------
    count = 0
    shift_flag = True
    print('=' * 80)
    if is_no_gra_loss:
        print('Testing with ablation model (No Gradient Loss):')
    elif is_cnn_fusion:
        print('Testing with ablation model (CNN Fusion instead of Cross-Attention):')
    else:
        print('Testing with standard trained model:')
    print('  Autoencoder IR: {}'.format(resume_model_auto_ir))
    print('  Autoencoder VI: {}'.format(resume_model_auto_vi))
    print('  Fusion Network: {}'.format(model_path_trans))
    print('  Test dataset: {}'.format(d_type))
    print('  Output path: {}'.format(output_path))
    print('=' * 80)
    with jt.no_grad():
        model_auto_ir, model_auto_vi, model_trans = load_model(custom_config_auto, custom_config_trans,
                                                            resume_model_auto_ir, resume_model_auto_vi,
                                                            model_path_trans, use_cnn_fusion=use_cnn_fusion_model)
        for ir_name in ir_names:
            if d_type.__contains__('21_pairs_tno') or d_type.__contains__('40_vot_tno'):
                vi_name = ir_name.replace('IR', 'VIS')
            else:
                vi_name = ir_name
            ir_path = os.path.join(test_path_ir, ir_name)
            vi_path = os.path.join(test_path_vi, vi_name)
            test(model_auto_ir, model_auto_vi, model_trans, shift_flag, ir_path, vi_path, ir_name, output_path, img_flag)
            print('Done. {}'.format(ir_name))
