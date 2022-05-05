import time, torch
from numpy.core.arrayprint import _none_or_positive_arg
import torchvision.models as models
from xautodl import config_utils
from xautodl.models import CellStructure, CellArchitectures, get_search_spaces
from xautodl.procedures import prepare_seed, get_optim_scheduler
from xautodl.procedures.starts import get_machine_info
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.config_utils import dict2config, load_config
from xautodl.log_utils import AverageMeter, Logger, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net
import timm

from pathlib import Path

from models.nas.xautodl.exps.NAS_Bench_201_algos.GDAS import GDAS
from models.nas.xautodl.exps.NAS_Bench_201_algos.DARTS_V1 import DARTS_V1
from models.nas.xautodl.exps.NAS_Bench_201_algos.DARTS_V2 import DARTS_V2
from models.nas.xautodl.exps.NAS_Bench_201_algos.ENAS import ENAS
from models.nas.xautodl.exps.NAS_Bench_201_algos.SETN import SETN
from models.nas.xautodl.exps.NATS_algos.search_cell import SearchCell
from models.nas.tenas.prune_tenas import TENAS
from models.nas.drnas.train_search import DrNAS
from models.nas.pc_darts.train_search import PC_DARTS
from models.nas.sdarts.train_search import SDARTS
from models.nas.tenas.evaluation.model import NetworkCIFAR as Network
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

manual_list=["resnet", "resnext", "wide_resnet", "vgg", "densenet", "efficientnet", "regnet", "cspnet", "bit", "dla"]
nas_list=["darts_v1", "darts_v2", "enas", "gdas", "tenas", "drnas", "pc_darts", "sdarts", "random", "setn"]

def procedure(xloader, network, criterion, scheduler, optimizer, mode, is_nas=True):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    if mode == "train":
        network.train()
    elif mode == "test":
        network.eval()
    else:
        raise ValueError("The mode is not right : {:}".format(mode))

    batch_time, end = AverageMeter(), time.time()
    for i, (inputs, targets) in enumerate(xloader):
        if mode == "train":
            scheduler.update(None, 1.0 * i / len(xloader))

        targets = targets.cuda(non_blocking=True)
        if mode == "train":
            optimizer.zero_grad()
        # forward
        if is_nas:
            features, logits = network(inputs)
        else:
            logits = network(inputs)
        loss = criterion(logits, targets)
        # backward
        if mode == "train":
            loss.backward()
            optimizer.step()
        # record loss and accuracy
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # count time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg, top1.avg, top5.avg, batch_time.sum


class GenericModel(object):
    """The generic model for various model architectures"""
    def __init__(self, args):
        super(GenericModel, self).__init__()
        self.args = args
        #self.arch=args.arch
        #self.model=self.get_model()
        

    def get_model(self,train_data,valid_data,xshape,class_num):
        arch=self.args.target_arch
        if arch in manual_list:
            # Human-designed models
            if arch=="resnet":
                model=timm.models.resnet.resnet18(in_chans=xshape[1],num_classes=class_num)
            elif arch=="resnext":
                model=timm.models.resnet.resnext50_32x4d(in_chans=xshape[1],num_classes=class_num)
            elif arch=="wide_resnet":
                model=timm.models.resnet.wide_resnet50_2(in_chans=xshape[1],num_classes=class_num)
            elif arch=="vgg":
                model=timm.models.vgg.vgg11_bn(in_chans=xshape[1],num_classes=class_num)
            elif arch=="densenet":
                model=timm.models.densenet.densenet121(in_chans=xshape[1],num_classes=class_num)
            elif arch=="efficientnet":
                model=timm.models.efficientnet.efficientnet_b0(in_chans=xshape[1],num_classes=class_num)
            elif arch=="regnet":
                model=timm.models.regnet.regnetx_004(in_chans=xshape[1],num_classes=class_num)
            elif arch=="cspnet":
                model=models.alexnet(num_classes=class_num)
                model=timm.models.cspnet.cspresnet50(in_chans=xshape[1],num_classes=class_num)
            elif arch=="bit":
                model=timm.models.resnetv2_50x1_bitm(in_chans=xshape[1],num_classes=class_num)
            elif arch=="dla":
                model=timm.models.dla.dla34(in_chans=xshape[1],num_classes=class_num) 
        elif arch in nas_list:
            # NAS-searched models
            arch_str=""
            model_args={
                    "data_path":self.args.data_path+"/"+self.args.dataset,
                    "dataset":self.args.dataset,
                    "search_space": "tss",
                    "use_api": 0,
                    "tau_min": 0.1,
                    "tau_max": 10,
                    "max_nodes": 4,
                    "channel": 16,
                    "num_cells": 5,
                    "eval_candidate_num": 100,
                    "track_running_stats": 0,
                    "affine": 0,
                    "config_path": "models/nas/xautodl/configs/nas-benchmark/algos/weight-sharing.config",
                    "overwite_epochs": None,
                    "arch_learning_rate": 1e-3,
                    "arch_weight_decay": 0,
                    "arch_eps": 1e-3,
                    "drop_path_rate": None,
                    "workers": 2,
                    "print_freq": 200,
                    "rand_seed": self.args.seed
                }
            
            # If the architecture is already searched by NAS
            # Just set direct_load = True to load it to save time
            direct_load=False

            if arch=="darts_v1":
                if direct_load:
                    arch_path=model_args["data_path"]+"/searched_archs"
                    file_path=arch_path+"/{}_{}_r_{}.txt".format("darts-v1",model_args["track_running_stats"],model_args["rand_seed"])
                    with open(file_path,"r",encoding="utf-8") as f:
                        arch_str=f.readline()
                    print("Architecture already generated in {}!\nThe searched architecture is {}".format(file_path,arch_str))
                else:
                    model_args["algo"]="darts-v1"
                    model_args["save_dir"]="./output/search/"+model_args["algo"]+"/"+self.args.dataset+"/latest"
                    if self.args.dataset=="stl10":
                        model_args["drop_path_rate"]=0.2
                    elif self.args.dataset=="cifar100":
                        model_args["drop_path_rate"]=0.1
                    elif self.args.dataset=="utkface":
                        model_args["drop_path_rate"]=0.3
                    arch_str=SearchCell(model_args,train_data,valid_data,xshape,class_num) 

            elif arch=="darts_v2":
                if direct_load:
                    arch_path=model_args["data_path"]+"/searched_archs"
                    file_path=arch_path+"/{}_{}_r_{}.txt".format("darts-v2",model_args["track_running_stats"],model_args["rand_seed"])
                    with open(file_path,"r",encoding="utf-8") as f:
                        arch_str=f.readline()
                    print("Architecture already generated in {}!\nThe searched architecture is {}".format(file_path,arch_str))
                else:
                    model_args["algo"]="darts-v2"
                    model_args["save_dir"]="./output/search/"+model_args["algo"]+"/"+self.args.dataset+"/latest"
                    if self.args.dataset=="stl10":
                        model_args["drop_path_rate"]=0.2
                    elif self.args.dataset=="cifar100":
                        model_args["drop_path_rate"]=0.1
                    elif self.args.dataset=="utkface":
                        model_args["drop_path_rate"]=0.3
                    arch_str=SearchCell(model_args,train_data,valid_data,xshape,class_num)
                
            elif arch=="enas":
                if direct_load:
                    arch_path=model_args["data_path"]+"/searched_archs"
                    file_path=arch_path+"/{}_{}_r_{}.txt".format("enas",model_args["track_running_stats"],model_args["rand_seed"])
                    with open(file_path,"r",encoding="utf-8") as f:
                        arch_str=f.readline()
                    print("Architecture already generated in {}!\nThe searched architecture is {}".format(file_path,arch_str))
                else:
                    model_args["algo"]="enas"
                    model_args["save_dir"]="./output/search/"+model_args["algo"]+"/"+self.args.dataset+"/latest"
                    if self.args.dataset=="stl10":
                        model_args["drop_path_rate"]=0.2
                    elif self.args.dataset=="cifar100":
                        model_args["drop_path_rate"]=0.1
                    arch_str=SearchCell(model_args,train_data,valid_data,xshape,class_num)

            elif arch=="gdas":
                if direct_load:
                    arch_path=model_args["data_path"]+"/searched_archs"
                    file_path=arch_path+"/{}_{}_r_{}.txt".format("gdas",model_args["track_running_stats"],model_args["rand_seed"])
                    with open(file_path,"r",encoding="utf-8") as f:
                        arch_str=f.readline()
                    print("Architecture already generated in {}!\nThe searched architecture is {}".format(file_path,arch_str))
                else:
                    model_args["algo"]="gdas"
                    model_args["save_dir"]="./output/search/"+model_args["algo"]+"/"+self.args.dataset+"/latest"
                    if self.args.dataset=="stl10":
                        model_args["arch_weight_decay"]=5e-4
                    arch_str=SearchCell(model_args,train_data,valid_data,xshape,class_num)

            elif arch=="setn":
                if direct_load:
                    arch_path=model_args["data_path"]+"/searched_archs"
                    file_path=arch_path+"/{}_{}_r_{}.txt".format("setn",model_args["track_running_stats"],model_args["rand_seed"])
                    with open(file_path,"r",encoding="utf-8") as f:
                        arch_str=f.readline()
                    print("Architecture already generated in {}!\nThe searched architecture is {}".format(file_path,arch_str))
                else:
                    model_args["algo"]="setn"
                    model_args["save_dir"]="./output/search/"+model_args["algo"]+"/"+self.args.dataset+"/latest"
                    arch_str=SearchCell(model_args,train_data,valid_data,xshape,class_num)

            elif arch=="random":
                if direct_load:
                    arch_path=model_args["data_path"]+"/searched_archs"
                    file_path=arch_path+"/{}_{}_r_{}.txt".format("random",model_args["track_running_stats"],model_args["rand_seed"])
                    with open(file_path,"r",encoding="utf-8") as f:
                        arch_str=f.readline()
                    print("Architecture already generated in {}!\nThe searched architecture is {}".format(file_path,arch_str))
                else:
                    model_args["algo"]="random"
                    model_args["save_dir"]="./output/search/"+model_args["algo"]+"/"+self.args.dataset+"/latest"
                    arch_str=SearchCell(model_args,train_data,valid_data,xshape,class_num)
            elif arch=="tenas":
                model_args={
                    "data_path": self.args.data_path+"/"+self.args.dataset,
                    "dataset":self.args.dataset,
                    "search_space_name": "darts",
                    "max_nodes": 4,
                    "track_running_stats": 0,
                    "workers": 2,
                    "batch_size": 16,
                    "rand_seed": self.args.seed,
                    "save_dir": "./output/search/tenas/"+self.args.dataset,
                    "arch_nas_dataset": None,
                    "precision": 3,
                    "prune_number": 3,
                    "repeat": 3,
                    "timestamp": "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time()))),
                    "init": "kaiming_normal",
                    "super_type": "nasnet-super",
                    "init_channels": 16,
                    "layers": 5,
                    "auxiliary": False
                }
                
                genotype_str=TENAS(model_args,train_data,valid_data,xshape,class_num,direct_load)
                model = Network(model_args["init_channels"], class_num, model_args["layers"], model_args["auxiliary"], eval(genotype_str))
                model.drop_path_prob = 0

            elif arch=="drnas":
                model_args={
                    "data_path": self.args.data_path+"/"+self.args.dataset,
                    "dataset":self.args.dataset,
                    "batch_size": 64,
                    "learning_rate": 0.1,
                    "learning_rate_min": 0.0,
                    "momentum": 0.9,
                    "weight_decay": 3e-4,
                    "report_freq": 50,
                    "init_channels": 16,
                    "layers": 5,
                    "cutout": False,
                    "cutout_length": 16,
                    "drop_path_prob": 0.3,
                    "save": "exp",
                    "seed": self.args.seed,
                    "grad_clip": 5,
                    "unrolled": False,
                    "arch_learning_rate": 6e-4,
                    "k": 4,
                    "reg_type": "l2",
                    "reg_scale": 1e-3,
                    "auxiliary": False
                }
                genotype_str = DrNAS(model_args,train_data,valid_data, xshape, class_num, direct_load)
                model = Network(model_args["init_channels"], class_num, model_args["layers"], model_args["auxiliary"], eval(genotype_str))
            
            elif arch=="pc_darts":
                model_args={
                    "data_path": self.args.data_path+"/"+self.args.dataset,
                    "dataset":self.args.dataset,
                    "batch_size": 64,
                    "learning_rate": 0.1,
                    "learning_rate_min": 0.0,
                    "momentum": 0.9,
                    "weight_decay": 3e-4,
                    "report_freq": 50,
                    "epochs": 50,
                    "init_channels": 16,
                    "layers": 5,
                    #"model_path": './output/',
                    "cutout": False,
                    "cutout_length": 16,
                    "drop_path_prob": 0.3,
                    "save": "exp",
                    "seed": self.args.seed,
                    "grad_clip": 5,
                    "unrolled": False,
                    "arch_learning_rate": 6e-4,
                    "arch_weight_decay": 1e-3,
                    "auxiliary": False
                }
                genotype_str = PC_DARTS(model_args,train_data,valid_data, xshape, class_num, direct_load)
                model = Network(model_args["init_channels"], class_num, model_args["layers"], model_args["auxiliary"], eval(genotype_str))

            elif arch=="sdarts":
                model_args={
                    "data_path": self.args.data_path+"/"+self.args.dataset,
                    "dataset":self.args.dataset,
                    "batch_size": 64,
                    "learning_rate": 0.025,
                    "learning_rate_min": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 3e-4,
                    "report_freq": 50,
                    "epochs": 50,
                    "init_channels": 16,
                    "layers": 5,
                    #"model_path": './output/',
                    "cutout": False,
                    "cutout_length": 16,
                    "cutout_prob": 1.0,
                    "drop_path_prob": 0.3,
                    "save": "exp",
                    "seed": self.args.seed,
                    "grad_clip": 5,
                    "unrolled": False,
                    "arch_learning_rate": 3e-4,
                    "arch_weight_decay": 1e-3,
                    "search_space": "s1",
                    "perturb_alpha": "pgd_linf",
                    "epsilon_alpha": 0.3,
                    "auxiliary": False
                }
                genotype_str = SDARTS(model_args,train_data,valid_data, xshape, class_num, direct_load)
                model = Network(model_args["init_channels"], class_num, model_args["layers"], model_args["auxiliary"], eval(genotype_str))

            print("Parameter settings:\n{}".format(model_args))
            if arch in ["tenas","drnas","pc_darts","sdarts"]:
                return model
            model=self.str2net(arch_str,class_num)
        else:
            print("{} is not available! Please select from {} and {}".format(arch, manual_list, nas_list))
            exit(-1)

        return model


    def str2net(self,arch_str,class_num):
        if arch_str in CellArchitectures:
            arch = CellArchitectures[arch_str]
            print("The model string is found in pre-defined architecture dict : {}".format(arch_str))
        else:
            try:
                arch = CellStructure.str2structure(arch_str)
            except:
                raise ValueError(
                    "Invalid model string : {:}. It can not be found or parsed.".format(
                        arch_str
                    )
                )
        prepare_seed(self.args.seed)  # random seed
        net = get_cell_based_tiny_net(
            dict2config(
                {
                    "name": "infer.tiny",
                    "C": self.args.channel,
                    "N": self.args.num_cells,
                    "genotype": arch,
                    "num_classes": class_num,
                },
                None,
            )
        )
        return net
