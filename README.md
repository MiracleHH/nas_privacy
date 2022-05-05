# On the Privacy Risks of Cell-Based NAS Architectures

This is the code implementation for the paper "On the Privacy Risks of Cell-Based NAS Architectures".

## Environment Setup

We use Python 3.9.7 for our experiments. The other dependicies are shown in [requirements.txt](./requirements.txt). Please also set up the environment for nasbench301 according to its official implementation in [NAS-Bench-301](https://github.com/automl/nasbench301) if you want to analyze data and train the regression model by yourself.

## Dataset Preparation

The NAS-Bench-301 dataset is downloaded from [NAS-Bench-301 Dataset v1.0](https://figshare.com/articles/dataset/NAS-Bench-301_Dataset_v1_0/13246952), please convert the original architecture samples in ConfigSpace format to Genotype format before using them. The converted architectures with top-5% (i.e., 2678) test accuracy have been shown in the `nas_eval/data/genos/origin/` folder. And the architectures after MIA (i.e., membership inference attack) demotion/promotion modifications are shown in the `nas_eval/data/genos/changed` folder.

## Quick Start

You can use the following example commands to realize corresponding goals. 

### Privacy Measurement of Different Architectures
Go into the `nas_eval` folder first.

Take the TENAS algorithm (a NAS method) as an example, we can use the following command to search for the most suitable architecture and evaluate its MIA (i.e., membership inference attack) performance with the \<Black-Box, Shadow\> setting on the CIFAR10 dataset.

```Shell
python main.py --gpu 0 --dataset cifar10 --data_path ./data --attr attr --target_arch tenas --seed 2333 --bn 0 --attack meminf_0
```

### Effectiveness of Cell Pattern Modifications
In the `nas_eval` folder, we can use the following command to evaluate the MIA performance of the **original** architecture with ID 132 under the \<White-Box, Partial\> setting on the CIFAR10 dataset:

```Shell
python meminf_eval.py --data_path ./data --dataset cifar10 --geno_path ./data/genos/origin --attack meminf_2 --gpu 0 --channel 16 --num_cells 5 --geno_id 132 --defense 0 --seed 2333
```
Furthermore, if you want to check the corresponding MIA performance **after** the Only-Reduction MIA demotion cell patterns with the modification budget as 4, you can use the following command:

```Shell
python test_change.py --data_path ./data --dataset cifar10 --geno_path ./data/genos/changed --attack meminf_2 --gpu 0 --channel 16 --num_cells 5 --geno_id 132 --seed 2333 --cell_type 1 --change_type 0 --defense 0 --budget 4 --bn 0
```

Here, the parameter `cell_type` represents `Only-Normal`, `Only-Reduction` and `Dual` modifications for 0, 1 and 2 respectively, while the parameter `change_type` means MIA demotion and MIA promotion for 0 and 1 respectively. You can also set the defense type by the parameter `defense` (0 means no defense), and the parameter `budget` stands for the modification budget for the cell pattern.


## Instruction to Re-Analyze the Cell Patterns

We can follow the following instructions to re-genreate all necessary data and re-analyze the hidden cell patterns.

### Collecting MIA Evaluation Results

You can use the `meminf_eval.py` file to re-evaluate the original MIA performance of all sampled architectures (ID ranges from 0 to 2677), and then collect all these evalaution results together for further analysis. Our evaluation results for all architectures are shown in the `nas_eval/data/genos/origin/meminf` folder where the `genotype.txt`, `cell_indices.txt` and `test_auc.txt` files represent the recordings for the corresponding original architectures, ID indices and test MIA AUC scores.

### Regression Model Training

We need to prepare the *Architecture-to-MIA* dataset for the regression model first. 

Go into the `nasbench301` folder and use `data_process.py` to  convert the aforementioned collected evaluation results into a training dataset consistent with NAS-Bench-301 in format:

```Shell
python data_process.py --data_path ../nas_eval/data/genos/origin/meminf --save_path ../nas_eval/data/genos/origin/meminf/converted
```

Then we can use the following command to train a GIN regression model based on the evaluation results:

```
python surrogate_models/fit_meminf.py --model gnn_gin --nasbench_data ../nas_eval/data/genos/origin/meminf --data_config_path surrogate_models/configs/data_configs/nb_301_meminf.json --log_dir experiments/surrogate_models 
```

### Operation Importance Calculation

We can use the following command to calculate the Operation Importance scores for the edges existing in the architectures with the largest 297 (set 1 to the parameter`largest`) and the smallest 303 (set 0 to the parameter `largest`) MIA AUC socres respectively:

```Shell
python data_generation/full_analyze.py --largest 0 --use_t 0 --model_path ./experiments/surrogate_models/gnn_gin/20220216-082831-6 --data_path ../nas_eval/data/genos/origin/meminf --save_path ./nas_eval/data/genos/origin/meminf/stats
```

### Cell Pattern Extraction

Use the following command to extract the cell patterns for MIA demotion and promotion on both normal and reduction cells:

```Shell
python data_generation/extract.py --data_path ./nas_eval/data/genos/origin/meminf/stats --k 8
```

### Cell Architecture Modifications

Please manually add the generated cell patterns to the `data_generation/full_change.py` file, then use the following command to apply desired modifications (e.g., Only-Reduction MIA demotion modifications) to the original architectures:

```
python data_generation/full_change.py --data_path ../nas_eval/data/genos/origin/meminf --save_path ../nas_eval/data/genos/changed --change_type 0 --cell_type 1 --budget 4
```
