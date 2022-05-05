from ast import arg
import os
from collections import namedtuple
import argparse
import json
import numpy as np
from time import time

from nasbench301.surrogate_models import utils
from nasbench301.api import SurrogateAPI

from ConfigSpace.read_and_write import json as cs_json

import nasbench301 as nb
from copy import deepcopy

OPS = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
       ]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def parse_args():

    parser = argparse.ArgumentParser(description="Analye the characteristics of cell patterns.")
    parser.add_argument('--model_path', nargs='?', default='./experiments/surrogate_models/gnn_gin/20220216-082831-6',
                        help='file path for the model file.')
    parser.add_argument('--data_path', nargs='?', default='../nas_eval/data/genos/origin/meminf',
                        help='file path for the architecture data.')
    parser.add_argument('--save_path', nargs='?', default='/workspace/program/NAS_privacy/results/new2',
                        help='save path for the evaluation results of various cell architectures.')
    parser.add_argument('--largest', type=int, default=1,choices=[0,1],
                        help='The instances with the smallest (0) or the largest (1) evaluation metrics will be sampled.')
    parser.add_argument('--use_t', type=int, default=1,choices=[0,1],
                        help='Whether to use the real AUC score of the target cell as the ground truth to calculate the operation importance.')
    return parser.parse_args()

def get_neighbors(genotype, cell_type, op_idx):
    _idx=(op_idx[0]-2)*2+op_idx[1]
    target_op=eval('genotype.'+cell_type)[_idx]
    neighbor_cells=[]
    for op in OPS:
        if op != target_op[0]:
            _genotype=deepcopy(genotype)
            eval('_genotype.'+cell_type)[_idx]=(op,target_op[1])
            neighbor_cells.append(_genotype)

    another_op=eval('genotype.'+cell_type)[_idx+1-2*(_idx%2)]
    for i in range(op_idx[0]):
        if i != target_op[1] and i != another_op[1]:
            _genotype=deepcopy(genotype)
            eval('_genotype.'+cell_type)[_idx]=(target_op[0],i)
            neighbor_cells.append(_genotype)
    
    return neighbor_cells



def get_op_importance(genotype, surrogate_model, op_imp_stats, auc_truth):
    for cell_type in ['normal', 'reduce']:
        for end_node in range(2,6):
            for i in range(2):
                op=eval('genotype.'+cell_type)[(end_node-2)*2+i]
                key = (op[1],end_node,op[0])
                if key in op_imp_stats[cell_type].keys():
                    _neighbor_cells=get_neighbors(genotype, cell_type, (end_node, i))
                    test_aucs=0.0
                    for cell in _neighbor_cells:
                        test_aucs += surrogate_model.predict(config=cell, representation="genotype", with_noise=True)
                    oi=test_aucs/len(_neighbor_cells)-auc_truth
                    op_imp_stats[cell_type][key].append(oi)
                    
    return op_imp_stats


def extract(data_path):
    cell_configs, mia_aucs=[], []
    with open(os.path.join(data_path,'genotype.txt'),'r',encoding='utf-8') as f1:
        genos=f1.readlines()
        with open(os.path.join(data_path,'test_auc.txt'),'r',encoding='utf-8') as f2:
            aucs=f2.readlines()
            for idx, geno in enumerate(genos):
                cell_configs.append(eval(geno))
                mia_aucs.append(eval(aucs[idx])*100)
    return cell_configs, mia_aucs


def get_freq_edge(data_path, largest=0, threshold=0):
    op_imp_stats={cell_type: {} for cell_type in ['normal', 'reduce']}
    _op_imp_stats={cell_type: {} for cell_type in ['normal', 'reduce']}
    geno_configs, meminf_aucs = extract(data_path)
    meminf_aucs=np.array(meminf_aucs)
    if largest==0:
        # Only analyze the architectures with the smallest AUC scores
        topK=(meminf_aucs<78).sum()
        indices = np.argpartition(meminf_aucs, topK-1)[:topK]
        indices=indices[np.argsort(meminf_aucs[indices])]
    elif largest==1:
        # Only analyze the architectures with the largest AUC scores
        topK=(meminf_aucs>84).sum()
        indices = np.argpartition(meminf_aucs, -topK)[-topK:]
        indices=indices[np.argsort(-meminf_aucs[indices])]
    else:
        # Analyze all data
        indices=list(range(len(meminf_aucs)))

    for cell_type in ['normal','reduce']:
        for idx in indices:
            ops = eval('geno_configs[{}].{}'.format(idx,cell_type))
            for i, op in enumerate(ops):
                key = (op[1], i/2+2, op[0])
                if key not in op_imp_stats[cell_type].keys():
                    op_imp_stats[cell_type][key]=1
                else:
                    op_imp_stats[cell_type][key]+=1
        for key, value in op_imp_stats[cell_type].items():
            # Only consider the edges whose frequency is larger than a threshold
            if value > threshold:
                _op_imp_stats[cell_type][key]=[]

    return _op_imp_stats
                    


if __name__ == '__main__':
    args=parse_args()
    print("Arguments:\n{}".format(args))

    start=time()
    
    print("==> Loading performance surrogate model...")
    ensemble_dir_performance = args.model_path
    print(ensemble_dir_performance)
    performance_model = nb.load_ensemble(ensemble_dir_performance)

    geno_configs, meminf_aucs = extract(args.data_path)
    

    meminf_aucs=np.array(meminf_aucs)
    if args.largest:
        topK=(meminf_aucs>84).sum()
        indices = np.argpartition(meminf_aucs, -topK)[-topK:]
        indices=indices[np.argsort(-meminf_aucs[indices])]
        stats_name='largest_{}'.format(topK)
    else:
        topK=(meminf_aucs<78).sum()
        indices = np.argpartition(meminf_aucs, topK)[:topK]
        indices=indices[np.argsort(meminf_aucs[indices])]
        stats_name='smallest_{}'.format(topK)
    
    print("There are {} data samples in total, and {} instances will be sampled to analyze operation importance.".format(len(meminf_aucs),topK))
    print("TopK metrics:\n{}".format(list(meminf_aucs[indices])))

    op_imp_stats = get_freq_edge(args.data_path, largest=args.largest, threshold=14)


    print("⏰⏰⏰ Start evaluating the Operation Importance metrics!")

    # Use the real AUC score or the predicted AUC socre of the target architecture as the ground truth
    if args.use_t:
        stats_name+='_truth'
    else:
        stats_name+='_ntruth'
    
    for i,idx in enumerate(indices):
        t1=time()
        if args.use_t:
            op_imp_stats = get_op_importance(geno_configs[idx], performance_model, op_imp_stats, meminf_aucs[idx])
        else:
            target_auc=performance_model.predict(config=geno_configs[idx], representation="genotype", with_noise=True)
            op_imp_stats = get_op_importance(geno_configs[idx], performance_model, op_imp_stats, target_auc)
        print("Sample # {} (0-{}) has been evaluated! [{:.2f}s] | [{:.2f}s]".format(i,topK-1, time()-t1,time()-start))
    
    print('-'*50)
    print("Edge Importance Statistics:\n{}\n".format(op_imp_stats))
    print('-'*50)

    with open(os.path.join(args.save_path,'edge_imp_stats_{}.txt'.format(stats_name)),'w',encoding='utf-8') as f:
        f.write(str(op_imp_stats))

    print("🎉🎉🎉 Finished! Time cost: {:.2f}s.".format(time()-start))
