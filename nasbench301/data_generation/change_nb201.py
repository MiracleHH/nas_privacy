import argparse
import numpy as np

modified_ops={
    'larger':'nor_conv_3x3',
    'smaller':'avg_pool_3x3'
}

prefer_ops={'skip_connect':0,'nor_conv_1x1':1,'nor_conv_3x3':2,'avg_pool_3x3':3,'none':4}


def parse_args():

    parser = argparse.ArgumentParser(description="Transfer the cell patterns to different search spaces.")
    parser.add_argument('--data_path', nargs='?', default='./data',
                        help='file path for the dataset.')
    parser.add_argument('--save_path', nargs='?', default='./data',
                        help='file path to save the changed data.')
    parser.add_argument('--change_type', type=int, default=1,choices=[0,1],
                        help='The change made to the target architecture to make the evaluation metric smaller (0) or larger (1).')    
                        
    return parser.parse_args()

def convert(arch,larger=True):
    if larger:
        target_op=modified_ops['larger']
    else:
        target_op=modified_ops['smaller']

    final_arch=''

    node_ops=arch.split('+')
    for i,ops in enumerate(node_ops):
        _ops=ops[1:-1].split('|')
        op_tuples=[s.split('~') for s in _ops]
        #print("op_tuples: {}".format(op_tuples))
        exist=False
        for t in op_tuples:
            if t[0]==target_op:
                exist=True
                break
        if exist:
            if i==0:
                final_arch=ops
            else:
                final_arch=final_arch+"+"+ops
        else:
            order=[prefer_ops[t[0]] for t in op_tuples]
            pos=np.array(order).argmin()
            op_tuples[pos][0]=target_op
            new_ops="|"
            for t in op_tuples:
                new_ops=new_ops+"{}~{}|".format(t[0],t[1])
            if i==0:
                final_arch=new_ops
            else:
                final_arch=final_arch+"+"+new_ops
    return final_arch


if __name__ == '__main__':
    args=parse_args()
    name='larger' if args.change_type else 'smaller'

    with open('{}/transferred_cifar10_nb201_{}_genos.txt'.format(args.save_path,name),'w') as f2:
        with open(args.data_path+'/cifar10_nb201_orig_genos.txt','r') as f:
            data=f.readlines()
            N=len(data)
            for i,line in enumerate(data):
                if i==N-1:
                    converted_arch=convert(line,args.change_type)
                    print('The {}-th arch is: {}'.format(i,line))
                else:
                    converted_arch=convert(line[:-1],args.change_type)
                    print('The {}-th arch is: {}'.format(i,line[:-1]))
                print("The {}-th converted arch is: {}".format(i,converted_arch))
                f2.write("{}\n".format(converted_arch))

