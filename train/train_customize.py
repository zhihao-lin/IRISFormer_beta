import torch
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
# import random
from tqdm import tqdm
import time
import os, sys, inspect
pwdpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from pathlib import Path
os.system('touch %s/models_def/__init__.py'%pwdpath)
os.system('touch %s/utils/__init__.py'%pwdpath)
os.system('touch %s/__init__.py'%pwdpath)
print('started.' + pwdpath)
PACNET_PATH = Path(pwdpath) / 'third-parties' / 'pacnet'
sys.path.insert(0, str(PACNET_PATH))
print(sys.path)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# from dataset_openroomsV4_total3d_matcls_ import openrooms, collate_fn_OR
from dataset_openrooms_OR_scanNetPose_light_20210928 import openrooms, collate_fn_OR
from dataset_openrooms_OR_scanNetPose_light_20210928_iiw import iiw, collate_fn_iiw
from dataset_fipt import DatasetFIPT
from dataset_customize import DatasetCustomize
# from dataset_openrooms_OR_scanNetPose_binary_tables_ import openrooms_binary
# from dataset_openrooms_OR_scanNetPose_pickle import openrooms_pickle
# from utils.utils_dataloader_binary import make_data_loader_binary


from torch.nn.parallel import DistributedDataParallel as DDP
from utils.config import cfg
from utils.bin_mean_shift import Bin_Mean_Shift
from utils.bin_mean_shift_3 import Bin_Mean_Shift_3
from utils.bin_mean_shift_N import Bin_Mean_Shift_N
from utils.comm import synchronize
from utils.utils_misc import *
from utils.utils_dataloader import make_data_loader
from utils.utils_training import reduce_loss_dict, check_save, print_gpu_usage, time_meters_to_string, find_free_port
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR

import utils.utils_config as utils_config
from utils.utils_envs import set_up_envs
from icecream import ic

from utils.utils_training import cycle

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--data_root', default=None, help='path to input images')
parser.add_argument('--task_name', type=str, default='tmp', help='task name (e.g. N1: disp ref)')
parser.add_argument('--task_split', type=str, default='train', help='train, val, test', choices={"train", "val", "test"})
# Fine tune the model
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the model')
parser.add_argument("--if_train", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_val", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_vis", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_save", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_overfit_val", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--if_overfit_train", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--epochIdFineTune', type=int, default = 0, help='the training of epoch of the loaded model')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for depth component')
parser.add_argument('--reconstWeight', type=float, default=10, help='the weight for reconstruction error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for the rendering' )
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# Rui
# Device
# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--master_port", type=str, default='8914')

# DEBUG
parser.add_argument('--debug', action='store_true', help='Debug eval')
parser.add_argument('--batch_size_override_vis', type=int, default=-1, help='')
parser.add_argument('--ifMatMapInput', action='store_true', help='using mask as additional input')
# parser.add_argument('--ifDataloaderOnly', action='store_true', help='benchmark dataloading overhead')
parser.add_argument('--if_cluster', action='store_true', help='if using cluster')
parser.add_argument('--cluster', type=str, default='kubectl', help='cluster name if if_cluster is True', choices={"kubectl", "nvidia", "ngc"})
parser.add_argument('--if_hdr_input_matseg', action='store_true', help='if using hdr images')
parser.add_argument('--eval_every_iter', type=int, default=2000, help='')
parser.add_argument('--save_every_iter', type=int, default=5000, help='')
parser.add_argument('--debug_every_iter', type=int, default=2000, help='')
parser.add_argument('--max_iter', type=int, default=-1, help='')
parser.add_argument('--invalid_index', type=int, default = 0, help='index for invalid aread (e.g. windows, lights)')

# Pre-training
parser.add_argument('--resume', type=str, help='resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp); [to continue the current task, use: resume]', default='NoCkpt')
parser.add_argument('--resumes_extra', type=str, help='list of extra resumed checkpoints; strings concat by #', default='NoCkpt')
parser.add_argument('--reset_latest_ckpt', action='store_true', help='remove latest_checkpoint file')
parser.add_argument('--reset_scheduler', action='store_true', help='')
parser.add_argument('--reset_lr', action='store_true', help='')
parser.add_argument('--reset_tid', action='store_true', help='')
parser.add_argument('--tid_start', type=int, default=-1)
parser.add_argument('--epoch_start', type=int, default=-1)
# debug
# parser.add_argument("--mini_val", type=str2bool, nargs='?', const=True, default=False)
# to get rid of
parser.add_argument('--test_real', action='store_true', help='')

parser.add_argument('--skip_keys', nargs='+', help='Skip those keys in the model', required=False)
parser.add_argument('--replaced_keys', nargs='+', help='Replace those keys in the model', required=False)
parser.add_argument('--replacedby', nargs='+', help='... to match those keys from ckpt. Must be in the same length as ``replace_leys``', required=False)
parser.add_argument("--if_save_pickles", type=str2bool, nargs='?', const=True, default=False)

parser.add_argument('--meta_splits_skip', nargs='+', help='Skip those keys in the model', required=False)

# for warm-up lr scheduler
parser.add_argument('--epochs', default=150, type=int)
 # Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
# parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
#                     help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

# parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
#                     help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
# parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
#                     help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# The training weight on IIW
parser.add_argument('--rankWeight', type=float, default=2.0, help='the weight of ranking')
parser.add_argument("--if_train_OR", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_train_IIW", type=str2bool, nargs='?', const=True, default=True)

parser.add_argument(
    "--config-file",
    default=os.path.join(pwdpath, "configs/config.yaml"),
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "params",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
parser.add_argument('--dir_output')

# The detail model setting
opt = parser.parse_args()
# print(opt)
# os.environ['MASETER_PORT'] = str(nextPort(int(opt.master_port)))
os.environ['MASETER_PORT'] = str(find_free_port())
cfg.merge_from_file(opt.config_file)
cfg = utils_config.merge_cfg_from_list(cfg, opt.params)
opt.cfg = cfg
opt.pwdpath = pwdpath

# >>>>>>>>>>>>> A bunch of modularised set-ups
# opt.gpuId = opt.deviceIds[0]
semseg_configs = utils_config.load_cfg_from_cfg_file(os.path.join(pwdpath, opt.cfg.MODEL_SEMSEG.config_file))
# semseg_configs = utils_config.merge_cfg_from_list(semseg_configs, opt.params)
opt.semseg_configs = semseg_configs

from utils.utils_envs import set_up_dist
handle = set_up_dist(opt)
synchronize()

from utils.utils_envs import set_up_folders
set_up_folders(opt)

from utils.utils_envs import set_up_logger
logger, writer = set_up_logger(opt)

opt.logger = logger
set_up_envs(opt)
opt.cfg.freeze()

if opt.is_master:
    ic(opt.cfg)
# <<<<<<<<<<<<< A bunch of modularised set-ups

# >>>>>>>>>>>>> MODEL AND OPTIMIZER
from models_def.model_joint_all import Model_Joint as the_model
from models_def.model_joint_all_ViT import Model_Joint_ViT as the_model_ViT
# build model
# model = MatSeg_BRDF(opt, logger)
if opt.cfg.MODEL_ALL.enable:
    model = the_model_ViT(opt, logger)
else:
    model = the_model(opt, logger)

if opt.distributed: # https://github.com/dougsouza/pytorch-sync-batchnorm-example # export NCCL_LL_THRESHOLD=0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(opt.device)
if opt.cfg.MODEL_BRDF.load_pretrained_pth:
    model.load_pretrained_MODEL_BRDF(
        if_load_encoder=opt.cfg.MODEL_BRDF.pretrained_if_load_encoder,
        if_load_decoder=opt.cfg.MODEL_BRDF.pretrained_if_load_decoder,
        if_load_Bs=opt.cfg.MODEL_BRDF.pretrained_if_load_Bs
    )
if opt.cfg.MODEL_SEMSEG.enable and opt.cfg.MODEL_SEMSEG.if_freeze:
    # model.turn_off_names(['UNet'])
    model.turn_off_names(['SEMSEG_Net'])
    model.freeze_bn_semantics()
if opt.cfg.MODEL_MATSEG.enable and opt.cfg.MODEL_MATSEG.if_freeze:
    model.turn_off_names(['MATSEG_Net'])
    model.freeze_bn_matseg()

model.print_net()

# set up optimizers
# optimizer = get_optimizer(model.parameters(), cfg.SOLVER)
# print(model.BRDF_Net, '===f=asdfas')

optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.lr)
if opt.cfg.SOLVER.method == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=cfg.SOLVER.lr, weight_decay=0.01)

# optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.lr, betas=(0.5, 0.999) )
if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.dual_lr:
    assert False
    # backbone_params = []
    # other_params = []
    # for k, v in model.named_parameters():
    #     if 'backbone' in k:
    #         backbone_params.append(v)
    #         if opt.is_master:
    #             print(k)
    #     else:
    #         other_params.append(v)
    # # my_list = ['BRDF_Net.pretrained.model.patch_embed.backbone']
    # # backbone_params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
    # # other_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))
    # optimizer_backbone = optim.Adam(backbone_params, lr=1e-5, betas=(0.5, 0.999) )
    # optimizer_others = optim.Adam(other_params, lr=1e-4, betas=(0.5, 0.999) )
    # if opt.cfg.SOLVER.method == 'adamw':
    #     optimizer_backbone = optim.AdamW(backbone_params, lr=1e-5, weight_decay=0.05)
    #     optimizer_others = optim.AdamW(other_params, lr=1e-4, weight_decay=0.05)

if opt.cfg.MODEL_BRDF.DPT_baseline.enable and opt.cfg.MODEL_BRDF.DPT_baseline.if_SGD:
    assert False, 'SGD disabled.'
    # optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.lr, momentum=0.9)


if opt.distributed:
    model = DDP(model, device_ids=[opt.rank], output_device=opt.rank, find_unused_parameters=True)

logger.info(red('Optimizer: '+type(optimizer).__name__))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50, cooldown=0, verbose=True, threshold_mode='rel', threshold=0.01)
if opt.cfg.SOLVER.if_warm_up:
    # https://github.com/microsoft/Cream/blob/2fb020852cb6ea77bb3409da5319891a132ac47f/iRPE/DeiT-with-iRPE/main.py
    from timm.scheduler import create_scheduler
    scheduler, _ = create_scheduler(opt, optimizer)
    # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.dual_lr:
    #     scheduler_backbone, _ = create_scheduler(opt, optimizer_backbone)
    #     scheduler_others, _ = create_scheduler(opt, optimizer_others)

# if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.yogo_lr:
#     optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.lr)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# <<<<<<<<<<<<< MODEL AND OPTIMIZER

ENABLE_MATSEG = opt.cfg.MODEL_MATSEG.enable
opt.bin_mean_shift_device = opt.device if opt.cfg.MODEL_MATSEG.embed_dims <= 4 else 'cpu'
# opt.batch_size_override_vis = -1
if ENABLE_MATSEG:
    if opt.cfg.MODEL_MATSEG.embed_dims > 2:
        opt.batch_size_override_vis = 1
# opt.batch_size_override_vis = -1 if (opt.bin_mean_shift_device == 'cpu' or not ENABLE_MATSEG) else 1
if opt.cfg.MODEL_MATSEG.embed_dims == 2:
    bin_mean_shift = Bin_Mean_Shift(device=opt.device, invalid_index=opt.invalid_index)
else:
    bin_mean_shift = Bin_Mean_Shift_N(embedding_dims=opt.cfg.MODEL_MATSEG.embed_dims, \
        device=opt.bin_mean_shift_device, invalid_index=opt.invalid_index, if_freeze=opt.cfg.MODEL_MATSEG.if_freeze)
opt.bin_mean_shift = bin_mean_shift

# >>>>>>>>>>>>> DATASET
from utils.utils_semseg import get_transform_semseg, get_transform_matseg, get_transform_resize

transforms_train_semseg = get_transform_semseg('train', opt)
transforms_val_semseg = get_transform_semseg('val', opt)
transforms_train_matseg = get_transform_matseg('train', opt)
transforms_val_matseg = get_transform_matseg('val', opt)
transforms_train_resize = get_transform_resize('train', opt)
transforms_val_resize = get_transform_resize('val', opt)

openrooms_to_use = openrooms
make_data_loader_to_use = make_data_loader

print('+++++++++openrooms_to_use', openrooms_to_use)
# <<<<<<<<<<<<< DATASET

# >>>>>>>>>>>>> DATASET
transforms_train_semseg_iiw = get_transform_semseg('train', opt)
transforms_val_semseg_iiw = get_transform_semseg('val', opt)
transforms_train_matseg_iiw = get_transform_matseg('train', opt)
transforms_val_matseg_iiw = get_transform_matseg('val', opt)
transforms_train_resize_iiw = get_transform_resize('train', opt)
transforms_val_resize_iiw = get_transform_resize('val', opt)

if opt.if_vis:
    iiw_dataset_val_vis = DatasetCustomize(opt,
        transforms_fixed = transforms_val_resize_iiw,
        transforms_semseg = transforms_val_semseg_iiw,
        transforms_matseg = transforms_val_matseg_iiw,
        transforms_resize = transforms_val_resize_iiw,
        cascadeLevel = opt.cascadeLevel, if_for_training=False, load_first = opt.cfg.TEST.vis_max_samples, logger=logger)
    iiw_loader_val_vis, batch_size_val_vis = make_data_loader(
        opt,
        iiw_dataset_val_vis,
        is_train=False,
        start_iter=0,
        logger=logger,
        workers=2,
        batch_size_override=opt.batch_size_override_vis,
        collate_fn=collate_fn_iiw,
        if_distributed_override=False
    )

from utils.utils_envs import set_up_checkpointing
checkpointer, tid_start, epoch_start = set_up_checkpointing(opt, model, optimizer, scheduler, logger)


# >>>>>>>>>>>>> TRANING
from train_funcs_joint_all import get_labels_dict_joint, val_epoch_joint, vis_val_epoch_joint, forward_joint, get_time_meters_joint
from train_funcs_joint_all_iiw import get_labels_dict_joint_iiw, val_epoch_joint_iiw, vis_val_epoch_joint_iiw, forward_joint_iiw, get_time_meters_joint_iiw
from train_funcs_dump import dump_joint

tid = tid_start

ts_iter_end_start_list = []
ts_iter_start_end_list = []
num_mat_masks_MAX = 0

model.train()
synchronize()

optimizer.zero_grad()


# for epoch in list(range(opt.epochIdFineTune+1, opt.cfg.SOLVER.max_epoch)):
# for epoch_0 in list(range(1, 2) ):
os.makedirs(opt.dir_output, exist_ok=True)
opt.summary_vis_path_task = opt.dir_output
val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid, 'bin_mean_shift': bin_mean_shift}
if opt.if_vis:
    val_params.update({'batch_size_val_vis': batch_size_val_vis})

    with torch.no_grad():
        vis_val_epoch_joint_iiw(iiw_loader_val_vis, model, val_params)
    synchronize()

