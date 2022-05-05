from copy import deepcopy
import numpy as np
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Extract the cell patterns.")
    parser.add_argument('--data_path', nargs='?', default='/workspace/program/NAS_privacy/data/subgraphs',
                        help='file path for the Operation Importance data.')
    parser.add_argument('--k', type=int, default=8,
                        help='The largest number of edges in the cell pattern.')
    return parser.parse_args()


if __name__ == '__main__':
    args=parse_args()
    print("Arguments:\n{}".format(args))


    cell_patterns={'larger':{'normal':[],'reduce':[]},
                'smaller':{'normal':[],'reduce':[]}}

    for change in ['larger','smaller']:
        print("="*30)
        print(change)
        print("-"*30)
        with open(args.data_path+"/edge_imp_stats_{}_ntruth.txt".\
                format('largest_297' if change=='larger' else 'smallest_303'),'r', encoding='utf-8') as f:
            data=f.readline()
            best_arch_op_imp=eval(data)
            for cell_type in ['normal','reduce']:
                print("{}:\nThere are {} different edges in total!".format(cell_type,len(best_arch_op_imp[cell_type])))
                sorted_edge_imp=sorted(best_arch_op_imp[cell_type].items(),key=lambda k:np.mean(k[1]),reverse=(change=='smaller'))

                exist_edges=[]
                node_entry={node:0 for node in range(2,6)}
                exist_nodes=[sorted_edge_imp[0][0][0]]
                while len(cell_patterns[change][cell_type])<args.k:
                    new_added=False
                    for item in sorted_edge_imp:
                        if item[0][:2] not in exist_edges and node_entry[item[0][1]]<2 and\
                        (item[0][0] in exist_nodes or item[0][1] in exist_nodes):
                            if (np.mean(item[1])<0 and change=='larger') or (np.mean(item[1])>0 and change=='smaller'):
                                print("{}: {}".format(item[0],np.mean(item[1])))
                                exist_edges.append(item[0][:2])
                                cell_patterns[change][cell_type].append(item[0])
                                node_entry[item[0][1]]+=1
                                if item[0][0] not in exist_nodes:
                                    exist_nodes.append(item[0][0])
                                if item[0][1] not in exist_nodes:
                                    exist_nodes.append(item[0][1])
                                sorted_edge_imp.remove(item)
                                new_added=True
                                break
                    if not new_added:
                        print("Too many required edges!")
                        break
                print("Cell Patterns:")
                for i,edge in enumerate(cell_patterns[change][cell_type]):
                    print("(\'{}\', {}, {}),".format(edge[2],edge[0],\
                                                            edge[1]))