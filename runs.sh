# vgg sparse 90%
python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.9 --end_step 0.5 --ramping True

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 127 --run_name 'run2_weight' \
--final_sparsity 0.9 --end_step 0.5 --ramping True

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 227 --run_name 'run3_weight' \
--final_sparsity 0.9 --end_step 0.5 --ramping True

# vgg sparse 95%
python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.95 --end_step 0.5 --ramping True

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 127 --run_name 'run2_weight' \
--final_sparsity 0.95 --end_step 0.5 --ramping True

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 227 --run_name 'run3_weight' \
--final_sparsity 0.95 --end_step 0.5 --ramping True

# vgg sparse 98%
python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.98 --end_step 0.5 --ramping True

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 127 --run_name 'run2_weight' \
--final_sparsity 0.98 --end_step 0.5 --ramping True

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 227 --run_name 'run3_weight' \
--final_sparsity 0.98 --end_step 0.5 --ramping True

# 90% sparse cyclic 
python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle'

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 127 --run_name 'run2_weight' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle'

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 227 --run_name 'run3_weight' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle'

## 95% 

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.95 --end_step 0.95 --ramping True --ramp_type 'half_cycle'

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 127 --run_name 'run2_weight' \
--final_sparsity 0.95 --end_step 0.95 --ramping True --ramp_type 'half_cycle'

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 227 --run_name 'run3_weight' \
--final_sparsity 0.95 --end_step 0.95 --ramping True --ramp_type 'half_cycle'

## 98% 


python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.98 --end_step 0.95 --ramping True --ramp_type 'half_cycle'

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 127 --run_name 'run2_weight' \
--final_sparsity 0.98 --end_step 0.95 --ramping True --ramp_type 'half_cycle'

python 'main.py' --model resnet32 --model_type sparse \
--lr 0.05 --dataset cifar100 --seed 227 --run_name 'run3_weight' \
--final_sparsity 0.98 --end_step 0.95 --ramping True --ramp_type 'half_cycle'
