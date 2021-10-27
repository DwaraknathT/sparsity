# sparsity
A codebase for research on sparse neural networks. Converting any model to make it ready for pruning is as simple as replacing all dense layers with masked dense layers and all conv layers with masked conv layers. 

## Union Mask 
Implements a simple technique called "Union Mask". The idea is to divide the entire training run into multiple cycles and use a cyclical sparsity scheduler along. For example, let's say total train steps are 100k, we divide them into 10 cycles each with with 10k steps. In each cycle, we start with 0% sparsity and prune the models to target sparsity. At the end of the cycle we save the masks and reset them to 0% sparsity. Before the final cycle, we logical or all the masks and prune the resultant mask, union mask, to target sparsity.  

## Some Interesting Results 
All results are an average of 3 independent runs on CIFAR 100. 

VGG19 
Dense Baseline - 72.9 
| Sparsity  | Vanilla | Union Mask |
| ------------- | ------------- | ------------- |
| 90 %  | 72.29  | 72.74 | 
| 95 % | 71.51  | 72.02 |
| 98 % | 67.48  | 69.34 |

ResNet 32 
Dense Baseline - 75.2 
| Sparsity  | Vanilla | Union Mask |
| ------------- | ------------- | ------------- |
| 90 %  |  74.09 | 74.52 | 
| 95 % | 72.79  | 73.45 |
| 98 % | 67.53  | 70.49 | 

## Features 
* Custom layers with parameter masks 
* Unstructured and structured pruning 
* Weight and unit pruning 
* [Single shot pruning before training (SNIP)](https://openreview.net/pdf?id=B1VZqjAcYX)
* [Single shot structured pruning before training](https://arxiv.org/abs/2007.00389)

## Pruning schedules available 
* Single shot pruning before training 
* Single shot pruning during training 
* Ramping pruning with linear ramping function
* Cyclical ramping pruning 

## Examples 
```
python main.py --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.9 --end_step 0.5 --ramping True
```
The above command trains a resnet 32 model on cifar100 with weights pruned in a ramping fashion, i.e, the networks starts with fully dense and sparsity is gradually increased to final sparisity level. Pruning stops once end step is reached. 

## Using Google Colab 
Its easy to use this repo with google colab 
* ```!git clone https://<access_token>@github.com/srk97/defense.git``` : Paste your GitHub access tokens without <>. You can get one from developer options in GitHub settings.
* ```!cd defense && git checkout <branch_name>``` : Do this only if you need to run on a branch
* Run the command. This will use your google drive to save the model under runs directory which can then be shared with others. You will see a prompt asking for a token when you execute the training script on colab.

## Acknowledgements 
* [Snip implementation in pytorch](https://github.com/mil-ad/snip)
