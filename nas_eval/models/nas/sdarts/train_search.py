import os
import sys
sys.path.insert(0, '../../')
import time
import glob
import numpy as np
import torch
import models.nas.sdarts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from models.nas.sdarts.model_search import Network
from models.nas.sdarts.architect import Architect
from models.nas.sdarts.spaces import spaces_dict

from models.nas.sdarts.attacker.perturb import Linf_PGD_alpha, Random_alpha

from copy import deepcopy
from numpy import linalg as LA

from torch.utils.tensorboard import SummaryWriter

'''

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--search_space', type=str, default='s1', help='searching space to choose from')
parser.add_argument('--perturb_alpha', type=str, default='none', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
args = parser.parse_args()


if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10
'''


def SDARTS(xargs, train_data, valid_data, xshape, class_num, direct_load=False):

    arch_path=xargs["data_path"]+"/searched_archs"
    if not os.path.exists(arch_path):
        os.mkdir(arch_path)
    file_path=arch_path+"/sdarts_0_{}_r_{}.txt".format(xargs["perturb_alpha"], xargs["seed"])
    if direct_load:
        with open(file_path,"r",encoding="utf-8") as f:
            genotype_str=f.readline()
        print("Architecture already generated in {}!\nThe searched architecture genotype is {}".format(file_path,genotype_str))
        return genotype_str

    xargs["save"] = './output/search/sdarts/{}/search-{}-{}-{}-{}'.format(
    xargs["dataset"], xargs["save"], time.strftime("%Y%m%d-%H%M%S"), xargs["search_space"], xargs["seed"])

    if xargs["unrolled"]:
        xargs["save"] += '-unrolled'
    if not xargs["weight_decay"] == 3e-4:
        xargs["save"] += '-weight_l2-' + str(xargs["weight_decay"])
    if not xargs["arch_weight_decay"] == 1e-3:
        xargs["save"] += '-alpha_l2-' + str(xargs["arch_weight_decay"])
    if xargs["cutout"]:
        xargs["save"] += '-cutout-' + str(xargs["cutout_length"]) + '-' + str(xargs["cutout_prob"])
    if not xargs["perturb_alpha"] == 'none':
        xargs["save"] += '-alpha-' + xargs["perturb_alpha"] + '-' + str(xargs["epsilon_alpha"])
    #xargs["save"] += '-' + str(np.random.randint(10000))

    utils.create_exp_dir(xargs["save"], scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(xargs["save"], 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    writer = SummaryWriter(xargs["save"] + '/runs')
    
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(xargs["seed"])
    #torch.cuda.set_device(xargs["gpu"])
    cudnn.benchmark = True
    torch.manual_seed(xargs["seed"])
    cudnn.enabled = True
    torch.cuda.manual_seed(xargs["seed"])
    #logging.info('gpu device = %d' % xargs["gpu"])
    logging.info("args = %s", xargs)

    if xargs["perturb_alpha"] == 'none':
        perturb_alpha = None
    elif xargs["perturb_alpha"] == 'pgd_linf':
        perturb_alpha = Linf_PGD_alpha
    elif xargs["perturb_alpha"] == 'random':
        perturb_alpha = Random_alpha

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(xargs["init_channels"], class_num, xargs["layers"], criterion, spaces_dict[xargs["search_space"]])
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        xargs["learning_rate"],
        momentum=xargs["momentum"],
        weight_decay=xargs["weight_decay"])

    '''
    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    if 'debug' in args.save:
        split = args.batch_size
        num_train = 2 * args.batch_size

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)
    '''
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=xargs["batch_size"],
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=xargs["batch_size"],
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(xargs["epochs"]), eta_min=xargs["learning_rate_min"])

    architect = Architect(model, xargs)

    for epoch in range(xargs["epochs"]):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        '''
        if args.cutout:
            # increase the cutout probability linearly throughout search
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)
        '''
        logging.info('epoch %d lr %e', epoch, lr)
        if xargs["perturb_alpha"]:
            epsilon_alpha = 0.03 + (xargs["epsilon_alpha"] - 0.03) * epoch / xargs["epochs"]
            logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(xargs, train_queue, valid_queue, model, architect, criterion, optimizer, lr, 
                                         perturb_alpha, epsilon_alpha)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)

        # validation
        valid_acc, valid_obj = infer(xargs, valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('Acc/valid', valid_acc, epoch)
        writer.add_scalar('Obj/valid', valid_obj, epoch)

        utils.save(model, os.path.join(xargs["save"], 'weights.pt'))

    genotype = model.genotype()
    genotype_str=str(genotype)
    logging.info('The final genotype is: %s', genotype_str)
    with open(file_path,"w") as f:
        f.writelines(genotype_str)
    writer.close()
    return genotype_str


def train(xargs, train_queue, valid_queue, model, architect, criterion, optimizer, lr, perturb_alpha, epsilon_alpha):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    #top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=xargs["unrolled"])
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        # print('before softmax', model.arch_parameters())
        model.softmax_arch_parameters()

        # perturb on alpha
        # print('after softmax', model.arch_parameters())
        if perturb_alpha:
            perturb_alpha(model, input, target, epsilon_alpha)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
        # print('after perturb', model.arch_parameters())

        logits = model(input, updateType='weight')
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), xargs["grad_clip"])
        optimizer.step()
        model.restore_arch_parameters()
        # print('after restore', model.arch_parameters())

        #prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        prec1 = utils.accuracy(logits, target, topk=(1,))[0]
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        #top5.update(prec5.data, n)

        if step % xargs["report_freq"] == 0:
            #logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)
            if 'debug' in xargs["save"]:
                break

    return  top1.avg, objs.avg


def infer(xargs, valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    #top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            #prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            prec1 = utils.accuracy(logits, target, topk=(1,))[0]
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            #top5.update(prec5.data, n)

            if step % xargs["report_freq"] == 0:
                #logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
                if 'debug' in xargs["save"]:
                    break

    return top1.avg, objs.avg


if __name__ == '__main__':
    #main() 
    pass
