from copy import deepcopy
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from time import time
import argparse
from model import GenericModel
from data import Dataset
from attacks.utils.train import model_training
from attacks.doctor.attrinf import *
from attacks.doctor.modinv import *
from attacks.doctor.modsteal import *
from attacks.doctor.meminf import *
from attacks.doctor.define_models import *
from attacks.utils.dcgan import *
from attacks.utils.train import *
from attacks.utils.dataloader import *
from models.nas.tenas.evaluation.model import NetworkCIFAR as Network
from collections import namedtuple
from opacus.validators.module_validator import ModuleValidator


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

nas_list=["darts_v1", "darts_v2", "enas", "gdas", "tenas", "drnas", "pc_darts", "sdarts", "random", "setn"]

def parse_args():

    parser = argparse.ArgumentParser(description="Meauring Privacy Risks of Machine Learning Models.")
    parser.add_argument('--data_path', nargs='?', default='./data',
                        help='file path for the dataset.')
    parser.add_argument('--dataset', nargs='?', default='cifar10',
                        help='Choose a dataset.')
    parser.add_argument('--geno_path', nargs='?', default='./results/new2',
                        help='The file path for cell genotypes.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--pretrain', nargs='?', default=None,
                        help='Whether to use pretrained models. If None, no pretrained models will be used.')
    parser.add_argument('--attack', nargs='?', default='meminf_0',
                        help='Specify an attack: meminf, attrinf, modelsteal.')
    parser.add_argument('--bn', type=int, default=0, choices=[0,1],
                        help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Manual random seed.')
    parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
    parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
    parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
    parser.add_argument('--geno_id', type=int, default=0, help='The ID for the selected genotype (0-9).')
    parser.add_argument('--attr', nargs='?', default='gender',
                        help='Specify the attribute(s) for attribute inference attacks.')
    parser.add_argument('--largest', type=int, default=1,choices=[0,1],
						help='The instances with the smallest (0) or the largest (1) evaluation metrics will be sampled.')
    parser.add_argument('--cell_type', type=int, default=1,choices=[0,1,2],
						help='The cell type of the target nas architecture to be changed. 0 for normal cell, 1 for reduce cell, and 2 for both of them.')
    parser.add_argument('--change_type', type=int, default=1,choices=[0,1],
						help='The change made to the target architecture to make the evaluation metric smaller (0) or larger (1).') 
    parser.add_argument('--maxnode', type=int, default=4,
						help='The maximum number of intermediate nodes in a cell.')
    parser.add_argument('--defense', type=int, default=0,
						help='The type of defense strategy.')
    parser.add_argument('--budget', type=int, default=3,
                        help='The budget for the cell patterns.')
    return parser.parse_args()

def data2np(data):
    X,y=[],[]
    for _X,_y in data:
        X.append(_X.numpy().tolist())
        y.append(_y)
    return np.array(X).astype(np.float32), np.array(y).astype(np.int8)

def train_model(PATH, device, train_set, test_set, model, use_DP, noise, norm, is_nas=True, use_label_smooth=False, pretrain=False):
    FILE_PATH = PATH + "_target.pth"
    if use_DP:
        errors = ModuleValidator.validate(model, strict=False)
        if len(errors)!=0:
            model=ModuleValidator.fix(model)
    if pretrain:
            load_wo_module(model,FILE_PATH)
            print("Directly loaded successfully!")
            return -1, -1, -1, model
    _model=deepcopy(model)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)
    
    train_model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, is_nas, use_label_smooth)
    acc_train = 0
    acc_test = 0

    for i in range(100):
        t1=time.time()
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")

        acc_train = train_model.train()
        print("target testing")
        acc_test = train_model.test()

        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)
        print("[{:.2f}s]".format(time.time()-t1))

    train_model.saveModel(FILE_PATH)
    print("Target model saved to {}!!!".format(FILE_PATH))
    print("Finished training!!!")

    return acc_train, acc_test, overfitting, _model


def train_DCGAN(PATH, device, train_set, name):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2)

    if name.lower() == 'fmnist':
        D = FashionDiscriminator(ngpu=1).eval()
        G = FashionGenerator(ngpu=1).eval()
    else:
        D = Discriminator(ngpu=1).eval()
        G = Generator(ngpu=1).eval()

    print("Starting Training DCGAN...")
    GAN = GAN_training(train_loader, D, G, device)
    for i in range(200):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        GAN.train()

    GAN.saveModel(PATH + "_discriminator.pth", PATH + "_generator.pth")


def test_meminf(PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, use_DP, noise, norm, attack_type, is_nas=True, use_memGuard=False, use_label_smooth=False):
    batch_size = 64

    nas_flag=0
    if is_nas:
        nas_flag+=1
        if attack_type != "meminf_0":
            nas_flag+=1

    attack_with_shadow, white_box_attacks=["meminf_0","meminf_3"], ["meminf_2", "meminf_3"]

    if attack_type in attack_with_shadow:
        # Train and save shadow model
        shadow_trainloader = torch.utils.data.DataLoader(
            shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
        shadow_testloader = torch.utils.data.DataLoader(
            shadow_test, batch_size=batch_size, shuffle=True, num_workers=2)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        acc_train, acc_test, overfitting, shadow_model=train_shadow_model(PATH, device, shadow_model, shadow_trainloader, shadow_testloader, use_DP, noise, norm, batch_size, loss, optimizer, nas_flag-1, use_label_smooth=use_label_smooth)

    if attack_type in white_box_attacks:
        #for white box
        gradient_size = get_gradient_size(target_model)
        total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2

    # Attack Evaluation
    if attack_type=="meminf_0":
        # Black-Box/Shadow
        attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, batch_size)

        attack_model = ShadowAttackModel(num_classes)
        attack_mode0(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes, nas_flag, use_memGuard)

    elif attack_type=="meminf_1":
        # Black-Box/Partial
        attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)

        attack_model = PartialAttackModel(num_classes)
        attack_mode1(PATH + "_target.pth", PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, 1, num_classes, nas_flag, use_memGuard)

    elif attack_type=="meminf_2":
        # White-Box/Partial
        attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)

        attack_model = WhiteBoxAttackModel(num_classes, total)
        attack_mode2(PATH + "_target.pth", PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, 1, num_classes, nas_flag, use_memGuard)

    elif attack_type=="meminf_3":
        # White-Box/Shadow
        attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, batch_size)

        attack_model = WhiteBoxAttackModel(num_classes, total)
        attack_mode3(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device,\
            attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes, nas_flag, use_memGuard)

    elif attack_type=="meminf_label_only":
        # Label-Only
        X_train, y_train=data2np(target_train)
        X_test, y_test=data2np(target_test)

        sample_size=1000
        label_only_attack(target_model, X_train, y_train, X_test, y_test, sample_size, X_train[0].shape, num_classes, 2333, is_nas)
        

def test_modinv(PATH, device, num_classes, target_train, target_model, name):
    size = (1,) + tuple(target_train[0][0].shape)
    target_model, evaluation_model = load_data(PATH + "_target.pth", PATH + "_eval.pth", target_model, models.resnet18(num_classes=num_classes))

    # CCS 15
    modinv_ccs = ccs_inversion(target_model, size, num_classes, 1, 3000, 100, 0.001, 0.003, device)
    train_loader = torch.utils.data.DataLoader(target_train, batch_size=1, shuffle=False)
    ccs_result = modinv_ccs.reverse_mse(train_loader)

    # Secret Revealer

    if name.lower() == 'fmnist':
        D = FashionDiscriminator(ngpu=1).eval()
        G = FashionGenerator(ngpu=1).eval()
    else:
        D = Discriminator(ngpu=1).eval()
        G = Generator(ngpu=1).eval()

    PATH_D = PATH + "_discriminator.pth"
    PATH_G = PATH + "_generator.pth"
    
    D, G, iden = prepare_GAN(name, D, G, PATH_D, PATH_G)
    modinv_revealer = revealer_inversion(G, D, target_model, evaluation_model, iden, device)

def test_attrinf(PATH, device, num_classes, target_train, target_test, target_model):
    attack_length = int(0.5 * len(target_train))
    rest = len(target_train) - attack_length

    attack_train, _ = torch.utils.data.random_split(target_train, [attack_length, rest])
    attack_test = target_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=64, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=64, shuffle=True, num_workers=2)

    image_size = [1] + list(target_train[0][0].shape)
    train_attack_model(
        PATH + "_target.pth", PATH, num_classes, device, target_model, attack_trainloader, attack_testloader, image_size)

def test_modsteal(PATH, device, train_set, test_set, target_model, attack_model):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)

    loss = nn.MSELoss()
    optimizer = optim.SGD(attack_model.parameters(), lr=0.01, momentum=0.9)

    attacking = train_steal_model(
        train_loader, test_loader, target_model, attack_model, PATH + "_target.pth", PATH + "_modsteal.pth", device, 64, loss, optimizer)

    for i in range(100):
        print("[Epoch %d/%d] attack training"%((i+1), 100))
        attacking.train_with_same_distribution()
    
    print("Finished training!!!")
    attacking.saveModel()
    acc_test, agreement_test = attacking.test()
    print("Saved Target Model!!!\nstolen test acc = %.3f, stolen test agreement = %.3f\n"%(acc_test, agreement_test))


if __name__=="__main__":
    args=parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = Dataset(args.dataset,args.data_path, args.attr)

    print("Arguments:\n{}".format(args))
    
    if args.dataset=="cifar100":
        # We need to ensure that each class is evenly distributed in every subset for CIFAR100 dataset
        # due to the small size (only 600) for each class in the original dataset.
        target_out, target_nas_tr, target_nas_val, shadow_train, shadow_out = data.get_iid_split_data(data.num_classes,seed=args.seed, augment=False)
    else:
        target_out, target_nas_tr, target_nas_val, shadow_train, shadow_out = data.get_split_data(seed=args.seed, augment=False)
    
    start=time.time()

    use_DP = False
    use_label_smooth=False
    use_memGuard=False
    noise=0.3
    norm = 1.5

    all_genos = []
    if args.change_type:
        save_path="./data/models/changed/{}_ncell_{}_{}_larger_{}_defense_{}_static_{}_wtopo_ori_sep_adj".format(args.dataset, args.num_cells, args.geno_id, args.cell_type,args.defense, args.budget)
        file_name='budget_{}_wtopo_ntruth_max_original_sep_adj_all_2678_larger_{}.txt'.format(args.budget,args.cell_type)
    else:
        save_path="./data/models/changed/{}_ncell_{}_{}_smaller_{}_defense_{}_static_{}_wtopo_ori_sep_adj".format(args.dataset, args.num_cells, args.geno_id, args.cell_type,args.defense,args.budget)
        file_name='budget_{}_wtopo_ntruth_max_original_sep_adj_all_2678_smaller_{}.txt'.format(args.budget,args.cell_type)
    
    print("Genotype file name: {}".format(file_name))

    with open(os.path.join(args.geno_path,file_name),'r',encoding='utf-8') as f:
        _genos=f.readlines()
        for line in _genos:
            all_genos.append(eval(line))
    target_geno=all_genos[args.geno_id]

    print("*"*50)
    print("Genotype ID: {}\nGenotype:\n{}".format(args.geno_id, target_geno))
    print("*"*50)

    target_model = Network(args.channel, data.num_classes, args.num_cells, False, target_geno)


    white_box_attacks=["meminf_2","meminf_3"]
    if args.attack in white_box_attacks:
        shadow_model = deepcopy(target_model)
    else:
        shadow_model = CNN(input_channel=data.xshape[1], num_classes=data.num_classes, xshape=data.xshape)
    
    
    is_nas=True

    if args.defense==1:
        # Data Augmentation
        if args.dataset=="cifar100":
            target_out_aug, target_nas_tr_aug, target_nas_val_aug, shadow_train_aug, shadow_out_aug = data.get_iid_split_data(data.num_classes,seed=args.seed,augment=True)
        else:
            target_out_aug, target_nas_tr_aug, target_nas_val_aug, shadow_train_aug, shadow_out_aug = data.get_split_data(seed=args.seed,augment=True)

        acc_train, acc_test, overfitting, target_model= train_model(save_path, device, target_nas_tr_aug+target_nas_val_aug, target_out, target_model, use_DP, noise, norm, is_nas, use_label_smooth, pretrain=False)
    else:
        if args.defense==2:
            # Differential Privacy
            use_DP = True
            print("noise: {}, norm: {}".format(noise, norm))
        elif args.defense==3:
            # MemGuard
            # Only applicable to BLACK-BOX scenarios!
            use_memGuard=True
        elif args.defense==4:
            # Label Smoothing
            use_label_smooth=True
        acc_train, acc_test, overfitting, target_model=train_model(save_path, device, target_nas_tr+target_nas_val, target_out, target_model, use_DP, noise, norm, is_nas, use_label_smooth, pretrain=False)

    if args.attack[:6]=="meminf":
        test_meminf(save_path, device, data.num_classes, target_nas_tr+target_nas_val, target_out, shadow_train, shadow_out, target_model, shadow_model, use_DP, noise, norm, args.attack, is_nas, use_memGuard, use_label_smooth)
        
        # Other attacks not checked, just modify the following codes accordingly if you need.
    elif args.attack[:7]=="attrinf":
        test_attrinf(save_path, device, data.num_classes, target_nas_tr+target_nas_val, target_out, target_model)
    elif args.attack[:8]=="modsteal":
        test_modsteal(save_path, device, shadow_train+shadow_out, target_out, target_model, shadow_model)
    elif args.attack[:6]=="modinv":
        train_DCGAN(save_path, device, shadow_out + shadow_train, args.dataset)
        test_modinv(save_path, device, data.num_classes,  target_nas_tr+target_nas_val, target_model, args.dataset)
    else:
        print("Unsupported attack!")

    
    print("🕙🕙🕙 Time cost: {:.2f}s".format(time.time()-start))

