from copy import deepcopy
import os
import argparse
from collections import namedtuple
import numpy as np

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


# Cell patterns
MODIFIED_SET={
    'larger':{
        'normal':[
            ('dil_conv_3x3', 2, 4),
            ('sep_conv_3x3', 0, 4),
            ('sep_conv_3x3', 0, 5),
            ('sep_conv_3x3', 0, 2),
            ('sep_conv_5x5', 0, 3),
            ('sep_conv_3x3', 2, 3),
            ('sep_conv_3x3', 2, 5),
            ('sep_conv_3x3', 1, 2)
            ],
        'reduce':[
            ('sep_conv_3x3', 0, 5),
            ('sep_conv_3x3', 0, 4),
            ('dil_conv_5x5', 1, 4),
            ('sep_conv_5x5', 0, 3),
            ('sep_conv_5x5', 0, 2),
            ('sep_conv_5x5', 1, 3),
            ('sep_conv_5x5', 1, 2),
            ('sep_conv_5x5', 1, 5)
        ]
    },
    'smaller':{
        'normal':[
            ('dil_conv_5x5', 1, 3),
            ('sep_conv_5x5', 3, 4),
            ('sep_conv_5x5', 3, 5),
            ('sep_conv_5x5', 2, 4),
            ('sep_conv_5x5', 4, 5)
        ],
        'reduce':[
            ('avg_pool_3x3', 1, 2),
            ('avg_pool_3x3', 0, 2),
            ('avg_pool_3x3', 1, 3),
            ('avg_pool_3x3', 0, 3),
            ('avg_pool_3x3', 1, 4),
            ('avg_pool_3x3', 4, 5),
            ('avg_pool_3x3', 0, 4)
        ]
    }
    }



def parse_args():

    parser = argparse.ArgumentParser(description="Meauring Privacy Risks of Machine Learning Models.")
    parser.add_argument('--data_path', nargs='?', default='../nas_eval/data/genos/origin/meminf',
                        help='file path for the log file.')
    parser.add_argument('--save_path', nargs='?', default='../nas_eval/data/genos/changed',
						help='save path for the evaluation results of various cell architectures.')
    parser.add_argument('--cell_type', type=int, default=1,choices=[0,1,2],
						help='The cell type of the target nas architecture to be changed. 0 for normal cell, 1 for reduce cell, and 2 for both of them.')
    parser.add_argument('--change_type', type=int, default=1,choices=[0,1],
						help='The change made to the target architecture to make the evaluation metric smaller (0) or larger (1).')
    parser.add_argument('--budget', type=int, default=0,
						help='The modification budget of the cell patterns.')    
    return parser.parse_args()


def extract(data_path):
    cell_configs, mia_aucs, cell_indices =[], [], []
    with open(os.path.join(data_path,'genotype.txt'),'r',encoding='utf-8') as f1:
        genos=f1.readlines()
        with open(os.path.join(data_path,'test_auc.txt'),'r',encoding='utf-8') as f2:
            aucs=f2.readlines()
            with open(os.path.join(data_path,'cell_indices.txt'),'r',encoding='utf-8') as f3:
                indices=f3.readlines()
                for idx, geno in enumerate(genos):
                    cell_configs.append(eval(geno))
                    mia_aucs.append(eval(aucs[idx])*100)
                    cell_indices.append(eval(indices[idx]))
    return cell_configs, mia_aucs, cell_indices


def change_budget(genotype,cell_type,modified_set,budget):
    geno=deepcopy(genotype)
    changed=False
    if budget==0:
        return geno

    ops=eval('geno.{}'.format(cell_type))

    for op in modified_set[cell_type][:budget]:
        _idx=(op[2]-2)*2
        if ops[_idx]!=op[:2] and ops[_idx+1]!=op[:2]:
            if (*ops[_idx],op[2]) in modified_set[cell_type][:budget]:
                ops[_idx+1]=op[:2]
            else:
                ops[_idx]=op[:2]
            changed=True

    if changed:
        print("The target {} cell has been changed!".format(cell_type))
    return geno

        

if __name__ == '__main__':
    args=parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    
    geno_configs, meminf_aucs, cell_indices=[], [], []
    geno_configs, meminf_aucs, cell_indices = extract(args.data_path)


    cell_indices=np.array(cell_indices)
    topK=len(cell_indices)

    indices = np.argpartition(cell_indices, topK-1)[:topK]
    indices=indices[np.argsort(cell_indices[indices])]
    stats_name='all_{}'.format(topK)
    
    if args.change_type:
        modified_set=MODIFIED_SET['larger']
        stats_name+='_larger'
    else:
        modified_set=MODIFIED_SET['smaller']
        stats_name+='_smaller'

    if args.cell_type==0:
        cell_type=['normal']
    elif args.cell_type==1:
        cell_type=['reduce']
    elif args.cell_type==2:
        cell_type=['normal','reduce']
    
    print("There are {} data samples in total, and {} instances will be sampled to analyze operation importance.".format(len(cell_indices),topK))

    
    with open(os.path.join(args.save_path,'budget_{}_wtopo_ntruth_max_original_sep_adj_{}_{}.txt'.format(args.budget, stats_name,args.cell_type)),'w',encoding='utf-8') as f:
        for i,idx in enumerate(indices):
            changed_geno=deepcopy(geno_configs[idx])
            for _type in cell_type:
                changed_geno=change_budget(changed_geno,_type,modified_set,args.budget)
            f.write('{}\n'.format(str(changed_geno)))

    print("Groundtruth index order:\n{}".format(list(cell_indices[indices])))
    

    print("🎉🎉🎉 Finished!")