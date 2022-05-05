import os
import sys
sys.path.insert(0, '../')
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from models.nas.tenas.evaluation.net2wider import configure_optimizer, configure_scheduler
from models.nas.drnas.model_search import Network
from models.nas.drnas.architect import Architect
import models.nas.tenas.evaluation.utils as utils

'''
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--k', type=int, default=6, help='init partial channel parameter')
#### regularization
parser.add_argument('--reg_type', type=str, default='l2', choices=['l2', 'kl'], help='regularization type')
parser.add_argument('--reg_scale', type=float, default=1e-3, help='scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2')
args = parser.parse_args()


CIFAR_CLASSES = 10
if args.dataset == 'cifar100':
    CIFAR_CLASSES = 100
'''

    
def DrNAS(xargs, train_data, valid_data, xshape, class_num, direct_load=False):

  arch_path=xargs["data_path"]+"/searched_archs"
  if not os.path.exists(arch_path):
      os.mkdir(arch_path)
  file_path=arch_path+"/drnas_0_r_{}.txt".format(xargs["seed"])
  if direct_load:
    with open(file_path,"r",encoding="utf-8") as f:
      genotype_str=f.readline()
    print("Architecture already generated in {}!\nThe searched architecture genotype is {}".format(file_path,genotype_str))
    return genotype_str

  xargs["save"] = './output/search/drnas/{}/search-progressive-{}-{}-{}'.format(
      xargs["dataset"], xargs["save"], time.strftime("%Y%m%d-%H%M%S"), xargs["seed"])
  xargs["save"] += '-init_channels-' + str(xargs["init_channels"])
  xargs["save"] += '-layers-' + str(xargs["layers"]) 
  xargs["save"] += '-init_pc-' + str(xargs["k"])
  utils.create_exp_dir(xargs["save"], scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(xargs["save"], 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(xargs["seed"])
  #torch.cuda.set_device(xargs["gpu"])
  cudnn.benchmark = True
  torch.manual_seed(xargs["seed"])
  cudnn.enabled=True
  torch.cuda.manual_seed(xargs["seed"])
  #logging.info('gpu device = %d' % xargs["gpu"])
  logging.info("args = %s", xargs)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(xargs["init_channels"], class_num, xargs["layers"], criterion, k=xargs["k"],
                  reg_type=xargs["reg_type"], reg_scale=xargs["reg_scale"])
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
    model.parameters(),
    xargs["learning_rate"],
    momentum=xargs["momentum"],
    weight_decay=xargs["weight_decay"])

  '''
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.dataset=='cifar100':
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

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

  architect = Architect(model, xargs)

  # configure progressive parameter
  epoch = 0
  ks = [6, 4]
  num_keeps = [7, 4]
  train_epochs = [2, 2] if 'debug' in xargs["save"] else [25, 25]
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(sum(train_epochs)), eta_min=xargs["learning_rate_min"])

  for i, current_epochs in enumerate(train_epochs):
    for e in range(current_epochs):
      lr = scheduler.get_lr()[0]
      logging.info('epoch %d lr %e', epoch, lr)

      genotype = model.genotype()
      logging.info('genotype = %s', genotype)
      model.show_arch_parameters()

      # training
      train_acc, train_obj = train(xargs, train_queue, valid_queue, model, architect, criterion, optimizer, lr, e)
      logging.info('train_acc %f', train_acc)

      # validation
      valid_acc, valid_obj = infer(xargs, valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
      
      epoch += 1
      scheduler.step()
      utils.save(model, os.path.join(xargs["save"], 'weights.pt'))
    
    if not i == len(train_epochs) - 1:
      model.pruning(num_keeps[i+1])
      # architect.pruning([model.mask_normal, model.mask_reduce])
      model.wider(ks[i+1])
      optimizer = configure_optimizer(optimizer, torch.optim.SGD(
        model.parameters(),
        xargs["learning_rate"],
        momentum=xargs["momentum"],
        weight_decay=xargs["weight_decay"]))
      scheduler = configure_scheduler(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(sum(train_epochs)), eta_min=xargs["learning_rate_min"]))
      logging.info('pruning finish, %d ops left per edge', num_keeps[i+1])
      logging.info('network wider finish, current pc parameter %d', ks[i+1])

  genotype = model.genotype()
  genotype_str=str(genotype)
  logging.info('genotype = %s', genotype_str)
  with open(file_path,"w") as f:
    f.writelines(genotype_str)
  model.show_arch_parameters()
  return genotype_str


def train(xargs, train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
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

    if epoch >= 10:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=xargs["unrolled"])
    optimizer.zero_grad()
    architect.optimizer.zero_grad()

    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), xargs["grad_clip"])
    optimizer.step()
    optimizer.zero_grad()
    architect.optimizer.zero_grad()

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

  return top1.avg, objs.avg


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
