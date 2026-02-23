# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project: CrossFuse
# @File: test_color_image
# @Time: 2023/3/9 15:27


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


def test(model_auto_ir, model_auto_vi, model_trans, shift_flag, ir_path, vi_path, ir_name, output_path,
         output_path_fea):

    ir_img = utils.get_train_images(ir_path, None, None, flag=False)
    vi_img, vi_cb, vi_cr = utils.get_test_images_color(vi_path, None, None, flag=img_flag)
    
    # ---------------------------------------------
    # outputs = model.reconsturce(ir_img, vi_img)
    ir_sh, ir_de = model_auto_ir(ir_img)
    vi_sh, vi_de = model_auto_vi(vi_img)
    outputs = model_trans(ir_de, ir_sh, vi_de, vi_sh, shift_flag)
    img_out = outputs['out']
    ir_self = outputs['ir_self']
    vi_self = outputs['vi_self']
    fuse_cross = outputs['fuse_cross']
    # # ---------------------------------------------
    
    # ---------------------------------------------
    path_out = output_path + '/results_crossfuse_'
    path_out_fea = output_path_fea + '/result_crossfuse_'
    utils.save_image_color(img_out, vi_cb, vi_cr, path_out + ir_name)
    utils.save_image(ir_self, path_out_fea + 'irself_' + ir_name)
    utils.save_image(vi_self, path_out_fea + 'viself_' + ir_name)
    utils.save_image(fuse_cross, path_out_fea + 'cross_' + ir_name)


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
    
    # 使用你自己训练的模型
    resume_model_auto_ir = "./models/my_trained_models/autoencoder/auto_encoder_epoch_4_ir.model"
    resume_model_auto_vi = "./models/my_trained_models/autoencoder/auto_encoder_epoch_4_vi.model"
    # 如果需要使用原始预训练模型，可以改为：
    # resume_model_auto_ir = "./models/autoencoder/auto_encoder_epoch_4_ir.model"
    # resume_model_auto_vi = "./models/autoencoder/auto_encoder_epoch_4_vi.model"
    
    # ============================================================
    # 三种测试情况（根据需要修改模型路径）：
    # 1. 标准训练：Cross-Attention + 完整损失 (L_pix + L_gra)
    # 2. 梯度损失消融：Cross-Attention + 仅像素损失 (L_pix)
    # 3. CNN Fusion消融：CNN Fusion + 完整损失 (L_pix + L_gra)
    # ============================================================
    
    # 情况1：标准训练模型
    # model_path_trans = "./models/my_trained_models/transfuse/20251220_140533/fusetrans_epoch_16.model"
    
    # 情况2：梯度损失消融模型（去掉gra loss）
    # model_path_trans = "./models/my_trained_models/ablation/no_gra_loss/20251221_033618/fusetrans_epoch_8.model"
    
    # 情况3：CNN Fusion消融模型（替换Cross-Attention）
    model_path_trans = "./models/my_trained_models/ablation/transfuse_cnn/20251221_001106/fusetrans_epoch_8.model"
    
    # 如果需要使用原始预训练模型，可以改为：
    # model_path_trans = "./models/transfuse/fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model"
    # ----------------------------------------------------
    img_flag = True

    test_path_ir = './images/M3FD_Fusion/ir'
    test_path_vi = './images/M3FD_Fusion/vis'
    
    # ----------------------------------------------------
    # 从模型路径中自动识别模型类型，并生成对应的输出文件夹名称
    # 格式说明：
    # - 标准训练：M3FD_Fusion_epoch{epoch}_{session_id}
    # - 梯度损失消融：M3FD_Fusion_no_gra_loss_epoch{epoch}_{session_id}
    # - CNN Fusion消融：M3FD_Fusion_cnn_fusion_epoch{epoch}_{session_id}
    
    is_no_gra_loss = 'ablation/no_gra_loss' in model_path_trans
    is_cnn_fusion = 'ablation/transfuse_cnn' in model_path_trans
    use_cnn_fusion_model = is_cnn_fusion  # 用于加载模型
    
    match = re.search(r'(\d{8}_\d{6})/fusetrans_epoch_(\d+)', model_path_trans)
    if match:
        session_id = match.group(1)  # 20251220_140533
        epoch_num = match.group(2)   # 16 或 8
        if is_no_gra_loss:
            # 梯度损失消融
            data_type = '/M3FD_Fusion_no_gra_loss_epoch' + epoch_num + '_' + session_id
        elif is_cnn_fusion:
            # CNN Fusion消融
            data_type = '/M3FD_Fusion_cnn_fusion_epoch' + epoch_num + '_' + session_id
        else:
            # 标准训练
            data_type = '/M3FD_Fusion_epoch' + epoch_num + '_' + session_id
    else:
        # 如果路径格式不匹配，使用默认格式
        if is_no_gra_loss:
            data_type = '/M3FD_Fusion_no_gra_loss_my_trained_transfuse'
        elif is_cnn_fusion:
            data_type = '/M3FD_Fusion_cnn_fusion_my_trained_transfuse'
        else:
            data_type = '/M3FD_Fusion_my_trained_transfuse'
 
    # test_path_ir = './images/vot/ir'
    # test_path_vi = './images/vot/vis'
    # data_type = '/vot_transfuse'
    
    ir_pathes, ir_names = utils.list_images_test(test_path_ir)
    # ---------------------------------------------------
    output_path1 = './output/crossfuse_test'
    if os.path.exists(output_path1) is False:
        os.mkdir(output_path1)
    output_path = output_path1 + data_type
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    output_path_fea = output_path + '/feature'
    if os.path.exists(output_path_fea) is False:
        os.mkdir(output_path_fea)
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
    print('  Test dataset: M3FD_Fusion')
    print('  Output path: {}'.format(output_path))
    print('=' * 80)
    with jt.no_grad():
        model_auto_ir, model_auto_vi, model_trans = load_model(custom_config_auto, custom_config_trans,
                                                               resume_model_auto_ir, resume_model_auto_vi,
                                                               model_path_trans, use_cnn_fusion=use_cnn_fusion_model)
        for ir_name in ir_names:
            vi_name = ir_name
            ir_path = os.path.join(test_path_ir, ir_name)
            vi_path = os.path.join(test_path_vi, vi_name)
            test(model_auto_ir, model_auto_vi, model_trans, shift_flag, ir_path, vi_path, ir_name, output_path,
                 output_path_fea)
            print('Done. {}'.format(ir_name))

