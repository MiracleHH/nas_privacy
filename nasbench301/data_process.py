import os
import argparse
from representations import convert_genotype_to_config
from api import fixed_hyperparameters
import json
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def parse_args():

    parser = argparse.ArgumentParser(description="Meauring Privacy Risks of Machine Learning Models.")
    parser.add_argument('--data_path', nargs='?', default='/workspace/program/NAS_privacy/results/new2',
                        help='file path for the log file.')
    parser.add_argument('--save_path', nargs='?', default='/workspace/program/NAS_privacy/results/new2/converted',
						help='save path for the evaluation results of various cell architectures.')
    return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Covert the evaluation results to the format of nb301 data file
    cell_configs, test_aucs, cell_indices = [], [], []

    with open(os.path.join(args.data_path,'genotype.txt'),'r',encoding='utf-8') as f1:
        genos=f1.readlines()
        with open(os.path.join(args.data_path,'test_auc.txt'),'r',encoding='utf-8') as f2:
            aucs=f2.readlines()
            with open(os.path.join(args.data_path,'cell_indices.txt'),'r',encoding='utf-8') as f3:
                indices=f3.readlines()
                for idx,geno in enumerate(genos):
                    with open(os.path.join(args.save_path,'results_{}.json'.format(eval(indices[idx]))),'w',encoding='utf-8') as f:
                        config_dict=convert_genotype_to_config(eval(geno))
                        results={"optimized_hyperparamater_config":{**fixed_hyperparameters, **config_dict},"info":[{"val_accuracy":eval(aucs[idx])*100}], "test_accuracy":eval(aucs[idx])*100, "runtime": 30187.60}
                        json.dump(results,f)
    
    print("Finished!")