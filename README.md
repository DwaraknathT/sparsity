# sparsity
A codebase for research on sparse neural networks. Converting any model to make it ready for pruning is as simple as replacing all dense layers with masked dense layers and all conv layers with masked conv layers.

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
