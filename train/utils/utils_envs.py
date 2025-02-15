from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from utils.utils_misc import *
from utils.comm import synchronize, get_rank
import os, sys
from utils.utils_misc import only1true
from icecream import ic
import os
from utils import transform
import nvidia_smi
import datetime

def set_up_envs(opt):
    assert opt.cluster in opt.cfg.PATH.cluster_names
    CLUSTER_ID = opt.cfg.PATH.cluster_names.index(opt.cluster)
    opt.if_pad = False

    assert opt.cfg.SOLVER.method in ['adam', 'adamw', 'zhengqin-lightnet']


    opt.cfg.PATH.root = opt.cfg.PATH.root_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.PATH.root_local
    if opt.if_cluster:
        # opt.cfg.TRAINING.MAX_CKPT_KEEP = -1
        opt.if_save_pickles = False

    if opt.cfg.DEBUG.if_test_real:
        opt.cfg.DEBUG.if_dump_perframe_BRDF = True
        opt.cfg.TEST.vis_max_samples = 20000

    if opt.if_cluster:
        opt.cfg.DEBUG.if_fast_BRDF_labels = False
        opt.cfg.DEBUG.if_fast_light_labels = False

    if opt.cfg.DEBUG.if_fast_BRDF_labels:
        opt.cfg.DATASET.dataset_path_local = opt.cfg.DATASET.dataset_path_local_fast_BRDF

    # if opt.cfg.DEBUG.if_fast_light_labels:
    if opt.cfg.DEBUG.if_test_real:
        opt.cfg.DEBUG.dump_BRDF_offline.path_root_local = '/home/ruizhu/Documents/Projects/semanticInverse/third_parties_outside/VirtualObjectInsertion/BRDF_offline'
    opt.cfg.DEBUG.dump_BRDF_offline.path_root = opt.cfg.DEBUG.dump_BRDF_offline.path_root_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DEBUG.dump_BRDF_offline.path_root_local
    if not opt.cfg.DEBUG.if_test_real:
        opt.cfg.DEBUG.dump_BRDF_offline.path_task = str(Path(opt.cfg.DEBUG.dump_BRDF_offline.path_root) / opt.cfg.DEBUG.dump_BRDF_offline.task_name)
    else:
        opt.cfg.DEBUG.dump_BRDF_offline.path_task = opt.cfg.DEBUG.dump_BRDF_offline.path_root
    if opt.cfg.DEBUG.dump_BRDF_offline.enable:
        Path(opt.cfg.DEBUG.dump_BRDF_offline.path_root).mkdir(exist_ok=True)
        Path(opt.cfg.DEBUG.dump_BRDF_offline.path_task).mkdir(exist_ok=True)
        if not opt.cfg.DEBUG.if_test_real:
            for meta_split in ['main_xml', 'main_xml1', 'mainDiffMat_xml', 'mainDiffLight_xml1', 'mainDiffMat_xml1', 'mainDiffLight_xml']:
                (Path(opt.cfg.DEBUG.dump_BRDF_offline.path_task) / meta_split).mkdir(exist_ok=True)

    if opt.cfg.DATASET.if_quarter and not opt.if_cluster:
        opt.cfg.DATASET.dataset_path_local = opt.cfg.DATASET.dataset_path_local_quarter
    opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_local
    opt.cfg.DATASET.dataset_path_binary = opt.cfg.DATASET.dataset_path_binary_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_binary_local
    opt.cfg.DATASET.dataset_path_mini_binary = opt.cfg.DATASET.dataset_path_mini_binary_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_mini_binary_local
    opt.cfg.DATASET.dataset_path_pickle = opt.cfg.DATASET.dataset_path_pickle_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_pickle_local
    opt.cfg.DATASET.dataset_path_mini_pickle = opt.cfg.DATASET.dataset_path_mini_pickle_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_mini_pickle_local

    opt.cfg.DATASET.png_path = opt.cfg.DATASET.png_path_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.png_path_local
    opt.cfg.DATASET.dataset_path_mini = opt.cfg.DATASET.dataset_path_mini_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_mini_local
    opt.cfg.DATASET.dataset_path_mini_binary = opt.cfg.DATASET.dataset_path_mini_binary_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.dataset_path_mini_binary_local
    opt.cfg.DATASET.matpart_path = opt.cfg.DATASET.matpart_path_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.matpart_path_local
    opt.cfg.DATASET.matori_path = opt.cfg.DATASET.matori_path_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.matori_path_local
    opt.cfg.DATASET.envmap_path = opt.cfg.DATASET.envmap_path_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.envmap_path_local

    opt.cfg.DATASET.iiw_path = opt.cfg.DATASET.iiw_path_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.iiw_path_local
    opt.cfg.DATASET.nyud_path = opt.cfg.DATASET.nyud_path_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.DATASET.nyud_path_local

    if opt.data_root is not None:
        opt.cfg.DATASET.dataset_path = opt.data_root

    if opt.cfg.PATH.total3D_lists_path_if_zhengqinCVPR:
        assert False, 'paths not correctly configured! (we use Zhengqins test set as val set, but they are in a different path (/eccv20dataset/DatasetNew_test) than the main dataset'
        opt.cfg.PATH.total3D_lists_path = opt.cfg.PATH.total3D_lists_path_zhengqinCVPR
    opt.cfg.DATASET.dataset_list = os.path.join(opt.cfg.PATH.total3D_lists_path, 'list')

    if opt.cfg.DATASET.mini:
        opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_mini
        if opt.cfg.DATASET.if_binary:
            opt.cfg.DATASET.dataset_path_binary = opt.cfg.DATASET.dataset_path_mini_binary
        if opt.cfg.DATASET.if_pickle:
            opt.cfg.DATASET.dataset_path_pickle = opt.cfg.DATASET.dataset_path_mini_pickle
        opt.cfg.DATASET.dataset_list = opt.cfg.DATASET.dataset_list_mini
    if opt.cfg.DATASET.tmp:
        opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_tmp
        opt.cfg.DATASET.dataset_list = opt.cfg.DATASET.dataset_list_tmp
        opt.cf.DATASET.dataset_if_save_space = False
    opt.cfg.DATASET.dataset_list = os.path.join(opt.cfg.PATH.root, opt.cfg.DATASET.dataset_list)

    print('======= DATASET.dataset_path ', opt.cfg.DATASET.dataset_path)

    if opt.if_cluster and opt.cluster=='ngc':
        opt.cfg.flush_secs = 120
        # opt.cfg.DATASET.binary = True
        if opt.cfg.DATASET.if_to_memory:
            if opt.cfg.DATASET.if_quarter:
                opt.cfg.DATASET.dataset_path_binary += '-quarter'
                opt.cfg.DATASET.dataset_path_pickle += '-quarter'
            if opt.cfg.DATASET.if_binary:
                opt.cfg.DATASET.dataset_path_binary = opt.cfg.DATASET.dataset_path_binary.replace('/datasets_mount', opt.cfg.DATASET.memory_path)
            if opt.cfg.DATASET.if_pickle:
                opt.cfg.DATASET.dataset_path_pickle = opt.cfg.DATASET.dataset_path_pickle.replace('/datasets_mount', opt.cfg.DATASET.memory_path)

        print('=======', opt.cfg.DATASET.dataset_path_binary)
        print('=======', opt.cfg.DATASET.dataset_path_pickle)


    opt.cfg.MODEL_SEMSEG.semseg_path = opt.cfg.MODEL_SEMSEG.semseg_path_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.MODEL_SEMSEG.semseg_path_local
    opt.cfg.PATH.semseg_colors_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.semseg_colors_path)
    opt.cfg.PATH.semseg_names_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.semseg_names_path)
    opt.cfg.PATH.total3D_colors_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.total3D_colors_path)
    opt.cfg.PATH.total3D_lists_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.total3D_lists_path)
    # opt.cfg.PATH.total3D_data_path = opt.cfg.PATH.total3D_lists_path.parent.parent
    opt.cfg.PATH.OR4X_mapping_catInt_to_RGB = [os.path.join(opt.cfg.PATH.root, x) for x in opt.cfg.PATH.OR4X_mapping_catInt_to_RGB]
    opt.cfg.PATH.OR4X_mapping_catStr_to_RGB = [os.path.join(opt.cfg.PATH.root, x) for x in opt.cfg.PATH.OR4X_mapping_catStr_to_RGB]
    opt.cfg.PATH.matcls_matIdG1_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.matcls_matIdG1_path)
    opt.cfg.PATH.matcls_matIdG2_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.matcls_matIdG2_path)

    # sys.path.insert(0, str(Path(opt.cfg.DATASET.swin_path)))
    # sys.path.insert(0, str(Path(opt.cfg.DATASET.swin_path) / 'mmseg/models'))
    # sys.path.insert(0, str(Path(opt.cfg.DATASET.swin_path) / 'mmseg/models/backbones'))


    # ===== data =====
    opt.cfg.DATA.data_read_list = [x for x in list(set(opt.cfg.DATA.data_read_list.split('_'))) if x != '']

    opt.if_pad = False
    opt.if_resize = False
    if opt.cfg.MODEL_BRDF.DPT_baseline.enable or opt.cfg.MODEL_LIGHT.DPT_baseline.enable:
        opt.cfg.DATA.if_pad_to_32x = True

    if opt.cfg.DATA.if_pad_to_32x:
        opt.cfg.DATA.load_masks = True
        im_width_pad_to = int(np.ceil(opt.cfg.DATA.im_width/32.)*32)
        im_height_pad_to = int(np.ceil(opt.cfg.DATA.im_height/32.)*32)
        im_pad_with = 0
        pad_option = opt.cfg.DATA.pad_option
        assert pad_option in ['const', 'reflect']
        opt.if_pad = True
        if not opt.cfg.DEBUG.if_test_real: # if True, should pad indeptly for each sample
            opt.pad_op = transform.Pad([im_height_pad_to, im_width_pad_to], padding_with=im_pad_with, pad_option=pad_option)
        else:
            opt.pad_op = None
        opt.cfg.DATA.im_width_padded_to = im_width_pad_to
        opt.cfg.DATA.im_height_padded_to = im_height_pad_to

        # if not opt.cfg.DEBUG.if_iiw: # if True, should pad indeptly for each sample
        #     im_width_pad_to = int(np.ceil(opt.cfg.DATA.iiw.im_width/32.)*32) # 512
        #     im_height_pad_to = int(np.ceil(opt.cfg.DATA.iiw.im_height/32.)*32) # 342 -> 352
        #     opt.pad_op_iiw = transform.Pad([im_height_pad_to, im_width_pad_to], padding_with=im_pad_with, pad_option=pad_option)
        # else:
        #     opt.pad_op_iiw = None

    if opt.cfg.DEBUG.if_nyud: # if True, should pad indeptly for each sample
        if opt.cfg.DATA.if_pad_to_32x:
            im_width_pad_to = int(np.ceil(opt.cfg.DATA.im_width/32.)*32) # 320
            im_height_pad_to = int(np.ceil(opt.cfg.DATA.im_height/32.)*32) # 256
            opt.pad_op_nyud = transform.Pad([im_height_pad_to, im_width_pad_to], padding_with=im_pad_with, pad_option=pad_option)
        else:
            opt.pad_op_nyud = None

    if opt.cfg.DATA.if_resize_to_32x:
        im_width_resize_to = int(np.ceil(opt.cfg.DATA.im_width/32.)*32)
        im_height_resize_to = int(np.ceil(opt.cfg.DATA.im_height/32.)*32)
        opt.if_resize = True
        if not opt.cfg.DEBUG.if_test_real: # if True, should pad indeptly for each sample
            opt.resize_op = transform.Resize_flexible((im_width_resize_to, im_height_resize_to))
        else:
            opt.resize_op = None

    # opt.cfg.DATA.load_brdf_gt = False
    if opt.cfg.DEBUG.if_test_real:
        opt.cfg.DATA.load_light_gt = False
        if not opt.cfg.DEBUG.if_load_dump_BRDF_offline:
            opt.cfg.DATA.data_read_list = ''
            opt.cfg.DATASET.if_no_gt_BRDF = True
        opt.cfg.DATASET.if_no_gt_light = True

    # if opt.cfg.DEBUG.if_iiw:
    #     opt.cfg.DATASET.if_no_gt_BRDF = True


    # ====== MODEL_ALL =====
    if opt.cfg.MODEL_ALL.ViT_baseline.if_share_pretrained_over_BRDF_modalities:
        assert opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities
    opt.cfg.MODEL_ALL.enable_list = [x for x in opt.cfg.MODEL_ALL.enable_list.split('_') if x != '']
    if opt.cfg.MODEL_ALL.enable:
        assert opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.depth.activation in ['tanh', 'relu']
        # if not opt.cfg.DEBUG.if_test_real:
        #     opt.cfg.DATA.load_brdf_gt = True
        if any(x in opt.cfg.MODEL_ALL.enable_list for x in ['al', 'ro', 'de', 'no'] if x != ''):
            opt.cfg.MODEL_BRDF.enable = True
            opt.cfg.MODEL_BRDF.enable_list = list(set(opt.cfg.MODEL_BRDF.enable_list_allowed) & set(opt.cfg.MODEL_ALL.enable_list))
        if 'li' in opt.cfg.DATA.data_read_list or ('axis' in opt.cfg.DATA.data_read_list and 'lamb' in opt.cfg.DATA.data_read_list and 'weight' in opt.cfg.DATA.data_read_list):
            if not opt.cfg.DEBUG.if_test_real:
                opt.cfg.DATA.load_light_gt = True
                # opt.cfg.MODEL_LIGHT.use_GT_brdf = True
            opt.cfg.MODEL_LIGHT.enable = True
        if 'li' in opt.cfg.MODEL_ALL.enable_list:
            opt.cfg.MODEL_LIGHT.enable = True

        assert not (opt.cfg.MODEL_LIGHT.if_align_rerendering_envmap and opt.cfg.MODEL_LIGHT.if_align_log_envmap) # cannot be true at the same time
        

    # ====== BRDF =====
    if isinstance(opt.cfg.MODEL_BRDF.enable_list, str):
        opt.cfg.MODEL_BRDF.enable_list = [x for x in opt.cfg.MODEL_BRDF.enable_list.split('_') if x != '']
    opt.cfg.MODEL_BRDF.loss_list = [x for x in opt.cfg.MODEL_BRDF.loss_list.split('_') if x != '']

    assert opt.cfg.MODEL_BRDF.depth_activation in ['sigmoid', 'relu', 'tanh', 'midas']
    assert opt.cfg.MODEL_BRDF.loss.depth.if_use_midas_loss or opt.cfg.MODEL_BRDF.loss.depth.if_use_Zhengqin_loss
    assert not(opt.cfg.MODEL_BRDF.loss.depth.if_use_midas_loss and opt.cfg.MODEL_BRDF.loss.depth.if_use_Zhengqin_loss)
    
    # ====== DPT =====
    opt.cfg.MODEL_LIGHT.DPT_baseline.dpt_hybrid = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid
    if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.DPT_baseline.enable:
        # opt.cfg.DATA.if_load_png_not_hdr = True
        assert opt.cfg.MODEL_BRDF.DPT_baseline.model in ['dpt_large', 'dpt_base', 'dpt_hybrid', 'swin']
        
        assert opt.cfg.MODEL_BRDF.DPT_baseline.modality in ['al', 'de', 'enabled']

        assert opt.cfg.DATA.if_pad_to_32x or opt.cfg.DATA.if_resize_to_32x
        assert not(opt.cfg.DATA.if_pad_to_32x and opt.cfg.DATA.if_resize_to_32x)

        assert opt.cfg.MODEL_BRDF.DPT_baseline.readout in ['ignore', 'add', 'project']

        assert opt.cfg.MODEL_BRDF.DPT_baseline.feat_fusion_method in ['sum', 'concat']

        if 'dpt_hybrid' in opt.cfg.MODEL_BRDF.DPT_baseline.model:
            opt.cfg.MODEL_BRDF.DPT_baseline.feat_proj_channels = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.feat_proj_channels
        elif 'dpt_large' in opt.cfg.MODEL_BRDF.DPT_baseline.model:
            opt.cfg.MODEL_BRDF.DPT_baseline.feat_proj_channels = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_large.feat_proj_channels
        # else:
        #     assert False, 'not supported yet'

    # ====== per-pixel lighting =====
    if opt.cfg.MODEL_LIGHT.enable:
        # opt.cfg.DATA.load_brdf_gt = True
        # if not opt.cfg.DEBUG.if_test_real:
            # opt.cfg.DATA.load_light_gt = True
            # if opt.cfg.DATA.load_light_gt:
            #     opt.cfg.DATA.data_read_list += 'al_no_de_ro'.split('_')
        if opt.cfg.MODEL_LIGHT.use_GT_brdf and opt.cfg.MODEL_BRDF.enable:
            opt.cfg.DATA.load_brdf_gt = True
            opt.cfg.MODEL_LIGHT.freeze_BRDF_Net = True
            opt.cfg.MODEL_BRDF.if_freeze = True

        if opt.cfg.MODEL_LIGHT.freeze_BRDF_Net:
            opt.cfg.MODEL_BRDF.if_freeze = True
        #     opt.cfg.MODEL_BRDF.enable = False
        #     opt.cfg.MODEL_BRDF.enable_list = ''
        #     opt.cfg.MODEL_BRDF.loss_list = ''
        # else:
        #     opt.cfg.MODEL_BRDF.enable = True
        #     opt.cfg.MODEL_BRDF.enable_list += 'al_no_de_ro'.split('_')
        #     opt.cfg.MODEL_BRDF.enable_BRDF_decoders = True
        #     if opt.cfg.MODEL_LIGHT.freeze_BRDF_Net:
        #         opt.cfg.MODEL_BRDF.if_freeze = True

        opt.cfg.MODEL_LIGHT.enable_list = opt.cfg.MODEL_LIGHT.enable_list.split('_')

    # ====== semseg =====
    if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable or opt.cfg.MODEL_SEMSEG.use_as_input or opt.cfg.MODEL_MATSEG.use_semseg_as_input:
        opt.cfg.DATA.load_brdf_gt = True
        opt.cfg.DATA.load_semseg_gt = True
        opt.semseg_criterion = nn.CrossEntropyLoss(ignore_index=opt.cfg.MODEL_SEMSEG.semseg_ignore_label)
        assert opt.cfg.MODEL_SEMSEG.pspnet_version in [50, 101]
        opt.semseg_configs.layers = 50 if opt.cfg.MODEL_SEMSEG.pspnet_version == 50 else 101
        if opt.cfg.MODEL_SEMSEG.wallseg_only:
            opt.cfg.MODEL_SEMSEG.semseg_classes = 1
            opt.semseg_configs.classes = 1
            if opt.cfg.MODEL_BRDF.enable_semseg_decoder:
                opt.semseg_configs.train_w = opt.cfg.DATA.im_width
                opt.semseg_configs.train_h = opt.cfg.DATA.im_height

    if opt.cfg.MODEL_MATSEG.enable or opt.cfg.MODEL_MATSEG.use_as_input:
        opt.cfg.DATA.load_matseg_gt = True
    
    if opt.cfg.MODEL_BRDF.enable_semseg_decoder and opt.cfg.MODEL_SEMSEG.enable:
        raise (RuntimeError("Cannot be True at the same time: opt.cfg.MODEL_BRDF.enable_semseg_decoder, opt.cfg.MODEL_SEMSEG.enable"))

    # ====== matcls =====
    if opt.cfg.MODEL_MATCLS.enable:
        opt.cfg.DATA.load_matcls_gt = True

    # ====== BRDF, cont. =====
    opt.cfg.MODEL_BRDF.enable_BRDF_decoders = len(opt.cfg.MODEL_BRDF.enable_list) > 0

    # ic(opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders)
    if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        # if not opt.cfg.DEBUG.if_test_real:
        # opt.cfg.DATA.load_brdf_gt = True
        opt.depth_metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        if not opt.cfg.MODEL_LIGHT.freeze_BRDF_Net:
            opt.cfg.MODEL_BRDF.loss_list += opt.cfg.MODEL_BRDF.enable_list

    # ===== check if flags are legal =====
    check_if_in_list(opt.cfg.DATA.data_read_list, opt.cfg.DATA.data_read_list_allowed)
    check_if_in_list(opt.cfg.MODEL_BRDF.enable_list, opt.cfg.MODEL_BRDF.enable_list_allowed)
    check_if_in_list(opt.cfg.MODEL_BRDF.loss_list, opt.cfg.MODEL_BRDF.enable_list_allowed)
    check_if_in_list(opt.cfg.MODEL_ALL.enable_list, opt.cfg.MODEL_ALL.enable_list_allowed)

    # extra BRDF net params
    opt.cfg.MODEL_BRDF.encoder_exclude = opt.cfg.MODEL_BRDF.encoder_exclude.split('_')

    # export
    opt.cfg.PATH.torch_home_path = opt.cfg.PATH.torch_home_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.PATH.torch_home_local
    os.system('export TORCH_HOME=%s'%opt.cfg.PATH.torch_home_path)

    # mis
    # if opt.cfg.SOLVER.if_test_dataloader:
    #     opt.cfg.SOLVER.max_epoch = 10
    opt.cfg.PATH.pretrained_path = opt.cfg.PATH.pretrained_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.PATH.pretrained_local
    opt.cfg.PATH.models_ckpt_path = opt.cfg.PATH.models_ckpt_cluster[CLUSTER_ID] if opt.if_cluster else opt.cfg.PATH.models_ckpt_local

    # dump
    if opt.cfg.DEBUG.if_dump_shadow_renderer:
        opt.cfg.DEBUG.if_dump_anything = True
        opt.if_vis = True

        opt.cfg.MODEL_LIGHT.load_pretrained_MODEL_BRDF = False
        opt.cfg.MODEL_LIGHT.load_pretrained_MODEL_LIGHT = False
        opt.cfg.MODEL_BRDF.use_scale_aware_depth = True
        opt.cfg.MODEL_BRDF.depth_activation = 'relu'
        opt.cfg.DATA.data_read_list += ['mesh', 'de']

    # extra loss weights
    opt.loss_weight_dict = {
    }
    
def check_if_in_list(list_to_check, list_allowed, module_name='Unknown Module'):
    if len(list_to_check) == 0:
        return
    if isinstance(list_to_check, str):
        list_to_check = list_to_check.split('_')
    list_to_check = [x for x in list_to_check if x != '']
    if not all(e in list_allowed for e in list_to_check):
        print(list_to_check, list_allowed)
        error_str = red('Illegal %s of length %d: %s'%(module_name, len(list_to_check), '_'.join(list_to_check)))
        raise ValueError(error_str)

def set_up_logger(opt):
    from utils.logger import setup_logger, Logger, printer
    import sys

    # === LOGGING
    sys.stdout = Logger(Path(opt.summary_path_task) / 'log.txt')
    # sys.stdout = Logger(opt.summary_path_task / 'log.txt')
    logger = setup_logger("logger:train", opt.summary_path_task, opt.rank, filename="logger_maskrcn-style.txt")
    logger.info(red("==[config]== opt"))
    logger.info(opt)
    logger.info(red("==[config]== cfg"))
    logger.info(opt.cfg)
    logger.info(red("==[config]== Loaded configuration file {}".format(opt.config_file)))
    # logger.info(red("==[opt.semseg_configs]=="))
    # logger.info(opt.semseg_configs)

    with open(opt.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        # logger.info(config_str)
    printer = printer(opt.rank, debug=opt.debug)

    if opt.is_master and 'tmp' not in opt.task_name and not opt.cfg.DEBUG.if_test_real:
        exclude_list = ['apex', 'logs_bkg', 'archive', 'train_cifar10_py', 'train_mnist_tf', 'utils_external', 'build/'] + \
            ['Summary', 'Summary_vis', 'Checkpoint', 'logs', '__pycache__', 'snapshots', '.vscode', '.ipynb_checkpoints', 'azureml-setup', 'azureml_compute_logs']
        # if opt.if_cluster:
        # copy_py_files(opt.pwdpath, opt.summary_vis_path_task_py, exclude_paths=[str(opt.SUMMARY_PATH), str(opt.CKPT_PATH), str(opt.SUMMARY_VIS_PATH)]+exclude_list)
        # os.system('cp -r %s %s'%(opt.pwdpath, opt.summary_vis_path_task_py / 'train'))
        # logger.info(green('Copied source files %s -> %s'%(opt.pwdpath, opt.summary_vis_path_task_py)))
        # folders = [f for f in Path('./').iterdir() if f.is_dir()]
        # for folder in folders:
        #     folder_dest = opt.summary_vis_path_task_py / folder.name
        #     if not folder_dest.exists() and folder.name not in exclude_list:
        #         os.system('cp -r %s %s'%(folder, folder_dest))
    synchronize()

    if opt.is_master:
        writer = SummaryWriter(opt.summary_path_task, flush_secs=opt.cfg.flush_secs)
        print(green('=====>Summary writing to %s'%opt.summary_path_task))
    else:
        writer = None
    # <<<< SUMMARY WRITERS

    return logger, writer

def set_up_folders(opt):
    from utils.global_paths import SUMMARY_PATH, SUMMARY_VIS_PATH, CKPT_PATH
    opt.SUMMARY_PATH, opt.SUMMARY_VIS_PATH, opt.CKPT_PATH = SUMMARY_PATH, SUMMARY_VIS_PATH, CKPT_PATH

    # >>>> SUMMARY WRITERS
    if opt.if_cluster:
        if opt.cluster == 'kubectl':
            opt.home_path = Path('/ruidata/dptInverse/') 

        opt.CKPT_PATH = opt.home_path / CKPT_PATH
        opt.SUMMARY_PATH = opt.home_path / SUMMARY_PATH
        # if opt.cluster == 'ngc':
        #     opt.SUMMARY_PATH = opt.home_path_tmp / SUMMARY_PATH
        #     opt.SUMMARY_PATH.mkdir(exist_ok=True)
        opt.SUMMARY_VIS_PATH = opt.home_path / SUMMARY_VIS_PATH

    if not opt.if_cluster or 'DATE' in opt.task_name:
        if opt.resume != 'resume':
            opt.task_name = get_datetime() + '-' + opt.task_name
        # else:
        #     opt.task_name = opt.resume
        # print(opt.cfg)
    #     opt.root = opt.cfg.PATH.root_local
    # else:
    #     opt.root = opt.cfg.PATH.root_cluster
    opt.summary_path_task = opt.SUMMARY_PATH / opt.task_name
    # if opt.cluster == 'ngc':
    #     opt.summary_path_all_task = opt.SUMMARY_PATH_ALL / opt.task_name
    opt.checkpoints_path_task = opt.CKPT_PATH / opt.task_name
    opt.summary_vis_path_task = opt.SUMMARY_VIS_PATH / opt.task_name
    opt.summary_vis_path_task_py = opt.summary_vis_path_task / 'py_files'

    save_folders = [opt.summary_path_task, opt.summary_vis_path_task, opt.summary_vis_path_task_py, opt.checkpoints_path_task, ]
    print('====%d/%d'%(opt.rank, opt.num_gpus), opt.checkpoints_path_task)

    if opt.is_master:
        for root_folder in [opt.SUMMARY_PATH, opt.CKPT_PATH, opt.SUMMARY_VIS_PATH]:
            if not root_folder.exists():
                root_folder.mkdir(exist_ok=True)
        if_delete = 'n'
        print(green(opt.summary_path_task), os.path.isdir(opt.summary_path_task))
        if os.path.isdir(opt.summary_path_task):
            if 'POD' in opt.task_name:
                print('====opt.summary_path_task exists! %s'%opt.summary_path_task)
                if opt.resume != 'resume':
                    raise RuntimeError('====opt.summary_path_task exists! %s; opt.resume: %s'%(opt.summary_path_task, opt.resume))
                if_delete = 'n'
                # opt.resume = opt.task_name
                # print(green('Resuming task %s'%opt.resume))

                if opt.reset_latest_ckpt:
                    os.system('rm %s'%(os.path.join(opt.checkpoints_path_task, 'last_checkpoint')))
                    print(green('Removed last_checkpoint shortcut for %s'%opt.resume))
            else:
                if opt.resume == 'NoCkpt':
                    if_delete = 'y'
                elif opt.resume == 'resume':
                    if_delete = 'n'
                else:
                    if_delete = input(colored('Summary path %s already exists. Delete? [y/n] '%opt.summary_path_task, 'white', 'on_blue'))
                    # if_delete = 'y'
            if if_delete == 'y':
                for save_folder in save_folders:
                    os.system('rm -rf %s'%save_folder)
                    print(green('Deleted summary path %s'%save_folder))
        for save_folder in save_folders:
            if not Path(save_folder).is_dir() and opt.rank == 0:
                Path(save_folder).mkdir(exist_ok=True)

    synchronize()


def set_up_dist(opt):
    # os.environ['MASETER_PORT'] = str(find_free_port())
    os.environ['OMP_NUM_THREADS'] = str(1)
    local_rank = int(os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else 0)
    opt.rank = local_rank

    # >>>> DISTRIBUTED TRAINING
    torch.manual_seed(opt.cfg.seed)
    np.random.seed(opt.cfg.seed)
    random.seed(opt.cfg.seed)

    opt.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.distributed = opt.num_gpus > 1
    if opt.distributed:
        torch.cuda.set_device(opt.rank)
        opt.process_group = torch.distributed.init_process_group(
            backend="nccl", rank=opt.rank, world_size=opt.num_gpus, init_method="env://", timeout=datetime.timedelta(seconds=5400))
        # synchronize()
    # device = torch.device("cuda" if torch.cuda.is_available() and not opt.cpu else "cpu")
    opt.device = 'cuda'
    opt.if_cuda = opt.device == 'cuda'
    # opt.rank = get_rank()
    opt.is_master = opt.rank == 0
    print('=++++++ opt.rank:', opt.rank, opt.is_master)

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(opt.rank)
    # <<<< DISTRIBUTED TRAINING
    return handle

def set_up_checkpointing(opt, model, optimizer, scheduler, logger):
    from utils.checkpointer import DetectronCheckpointer

    # >>>> CHECKPOINTING
    save_to_disk = opt.is_master
    checkpointer = DetectronCheckpointer(
        opt, model, optimizer, scheduler, opt.CKPT_PATH, opt.checkpoints_path_task, save_to_disk, logger=logger, if_reset_scheduler=opt.reset_scheduler
    )
    tid_start = 0
    epoch_start = 0

    if opt.resume != 'NoCkpt':
        print('=+++++=opt.resume', opt.resume)
        if opt.resume == 'resume':
            opt.resume = opt.task_name
        replace_kws = []
        replace_with_kws = []
        if opt.replaced_keys is not None and opt.replacedby is not None:
            assert len(opt.replaced_keys) == len(opt.replacedby)
            replace_kws += opt.replaced_keys
            replace_with_kws += opt.replacedby
        # if opt.task_split == 'train':
        # if 'train_POD_matseg_Dd' in opt.resume:
        #     replace_kws = ['hourglass_model.seq_L2.1', 'hourglass_model.seq_L2.3', 'hourglass_model.disp_res_pred_layer_L2']
        #     replace_with_kws = ['hourglass_model.seq.1', 'hourglass_model.seq.3', 'hourglass_model.disp_res_pred_layer']
        checkpoint_restored, _, _ = checkpointer.load(task_name=opt.resume, skip_kws=opt.skip_keys if opt.skip_keys is not None else [], replace_kws=replace_kws, replace_with_kws=replace_with_kws)
    
        if opt.resumes_extra != 'NoCkpt':
            resumes_extra_list = opt.resumes_extra.split('#')
            for resume_extra in resumes_extra_list:
                _, _, _ = checkpointer.load(task_name=resume_extra, skip_kws=opt.skip_keys if opt.skip_keys is not None else [], replace_kws=replace_kws, replace_with_kws=replace_with_kws, prefix='[RESUME EXTRA] ')

        if 'iteration' in checkpoint_restored and not opt.reset_tid:
            tid_start = checkpoint_restored['iteration']
        if 'epoch' in checkpoint_restored and not opt.reset_tid:
            epoch_start = checkpoint_restored['epoch']
        if opt.tid_start != -1 and opt.epoch_start != -1:
            tid_start = opt.tid_start
            epoch_start = opt.epoch_start
        print(checkpoint_restored.keys())
        logger.info(colored('Restoring from epoch %d - iter %d'%(epoch_start, tid_start), 'white', 'on_blue'))

    if opt.reset_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.cfg.SOLVER.lr
            
    # <<<< CHECKPOINTING
    return checkpointer, tid_start, epoch_start