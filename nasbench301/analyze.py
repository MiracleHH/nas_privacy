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
from representations import OPS, Genotype
from copy import deepcopy

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def parse_args():

    parser = argparse.ArgumentParser(description="Analye the characteristics of cell patterns.")
    parser.add_argument('--model_path', nargs='?', default='./experiments/surrogate_models/gnn_gin/20220216-082831-6',
                        help='file path for the model file.')
    parser.add_argument('--save_path', nargs='?', default='/workspace/program/NAS_privacy/results/new2',
						help='save path for the evaluation results of various cell architectures.')
    parser.add_argument('--top_ratio', type=float, default=0.1,
						help='The ratio of total instances to be sampled with the largest or the smallest evaluation metrics.')
    parser.add_argument('--largest', type=int, default=1,choices=[0,1],
						help='The instances with the smallest (0) or the largest (1) evaluation metrics will be sampled.')
    parser.add_argument('--use_t', type=int, default=1,choices=[0,1],
						help='Whether to use the gound truth of the target cell to calculate the operation importance.')
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
    #print('Length of neighbor_cells: {}'.format(len(neighbor_cells)))
    #print("neighbor_cells[0]:\n{}".format(neighbor_cells[0]))
    return neighbor_cells

'''
def get_op_importance(genotype, surrogate_model):
    op_imp_stats={cell_type: {op:[] for op in OPS} for cell_type in ['normal', 'reduce']}
    target_acc=surrogate_model.predict(config=genotype_config, representation="genotype", with_noise=True)
    for cell_type in ['normal', 'reduce']:
        for end_node in range(2,6):
            for i in range(2):
                _neighbor_cells=get_neighbors(genotype, cell_type, (end_node, i))
                print("There are {} neighbor cells in total!".format(len(_neighbor_cells)))
                test_accs=0.0
                for cell in _neighbor_cells:
                    #print('🎉🎉🎉 cell:\n{}'.format(cell))
                    test_accs += surrogate_model.predict(config=cell, representation="genotype", with_noise=True)
                oi=test_accs/len(_neighbor_cells)-target_acc
                op_imp_stats[cell_type][eval('genotype.'+cell_type)[(end_node-2)*2+i][0]].append(oi)
    return op_imp_stats
'''
def get_op_importance(genotype, surrogate_model, op_imp_stats, auc_truth):
    total_neighbors=0
    for cell_type in ['normal', 'reduce']:
        for end_node in range(2,6):
            for i in range(2):
                _neighbor_cells=get_neighbors(genotype, cell_type, (end_node, i))
                test_aucs=0.0
                for cell in _neighbor_cells:
                    #print('🎉🎉🎉 cell:\n{}'.format(cell))
                    test_aucs += surrogate_model.predict(config=cell, representation="genotype", with_noise=True)
                oi=test_aucs/len(_neighbor_cells)-auc_truth
                op_imp_stats[cell_type][eval('genotype.'+cell_type)[(end_node-2)*2+i][0]].append(oi)
                total_neighbors+=len(_neighbor_cells)
    print("There are {} neighbor cells in total!".format(total_neighbors))
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
                    


if __name__ == '__main__':
    args=parse_args()
    print("Arguments:\n{}".format(args))

    start=time()
    # Load the performance surrogate model
    #NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
    #NOTE: Defaults to using the default model download path
    print("==> Loading performance surrogate model...")
    ensemble_dir_performance = args.model_path
    print(ensemble_dir_performance)
    performance_model = nb.load_ensemble(ensemble_dir_performance)

    '''
    data_config = json.load(open(os.path.join(ensemble_dir_performance, 'data_config.json'), 'r'))
    model_config = json.load(open(os.path.join(ensemble_dir_performance, 'model_config.json'), 'r'))
    
    surrogate_model = utils.model_dict['gnn_gin'](data_root='None', log_dir=ensemble_dir_performance, seed=data_config["seed"],
                                                  model_config=model_config, data_config=data_config)
    surrogate_model.load(model_path=ensemble_dir_performance+'/surrogate_model.model')
    performance_model = SurrogateAPI(surrogate_model)
    '''

    geno_configs, meminf_aucs=[], []
    for path in [args.save_path, args.save_path + '/meminf']:
        _geno_configs, _meminf_aucs=extract(path)
        geno_configs.extend(_geno_configs)
        meminf_aucs.extend(_meminf_aucs)
    
    #topK=round(len(meminf_aucs)*args.top_ratio)
    

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

    '''
    topk_aucs=meminf_aucs[indices]
    topk_genos=[]
    for idx in indices:
        topk_genos.append(geno_configs[idx])
    '''

    print("🧨🧨🧨 Start evaluating the Operation Importance metric!")
    if args.use_t:
        stats_name+='_truth'
    else:
        stats_name+='_ntruth'
    
    op_imp_stats={cell_type: {op:[] for op in OPS} for cell_type in ['normal', 'reduce']}
    for i,idx in enumerate(indices):
        t1=time()
        if args.use_t:
            op_imp_stats = get_op_importance(geno_configs[idx], performance_model, op_imp_stats, meminf_aucs[idx])
        else:
            target_auc=performance_model.predict(config=geno_configs[idx], representation="genotype", with_noise=True)
            op_imp_stats = get_op_importance(geno_configs[idx], performance_model, op_imp_stats, target_auc)
        print("Sample # {} (0-{}) has been evaluated! [{:.2f}s] | [{:.2f}s]".format(i,topK-1, time()-t1,time()-start))
    
    
    with open(os.path.join(args.save_path,'op_imp_stats_{}.json'.format(stats_name)),'w',encoding='utf-8') as f:
        json.dump(op_imp_stats,f)

    print("🎉🎉🎉 Finished! Time cost: {:.2f}s.".format(time()-start))

    '''

    # Option 1: Create a DARTS genotype
    print("==> Creating test configs...")
    genotype_config = Genotype(
            normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
            normal_concat=[2, 3, 4, 5],
            reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
            reduce_concat=[2, 3, 4, 5]
            )

    # Predict
    print("==> Predict runtime and performance...")
    prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=True)

    print("Genotype architecture performance: %f" %(prediction_genotype))

    print("Strat evaluating the target genotype!")

    op_imp_stats=get_op_importance(genotype_config,performance_model)
    print('🎉🎉🎉 op_imp_stats:\n{}'.format(op_imp_stats))
    '''