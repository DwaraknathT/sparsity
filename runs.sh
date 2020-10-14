# Train dense baselines using swa learning rate
python 'sparsity/main.py' --model vgg19_bn --model_type dense --use_colab True \
--lr 0.05 --dataset cifar100 --run_name 'run1'

python 'sparsity/main.py' --model vgg19_bn --model_type dense --use_colab True \
--lr 0.05 --dataset cifar100 --run_name 'run2' --seed 128

python 'sparsity/main.py' --model vgg19_bn --model_type dense --use_colab True \
--lr 0.05 --dataset cifar100 --run_name 'run3' --seed 3147

# vgg sparse 90%
python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.9 --end_step 0.5 --ramping True

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --seed 128 --run_name 'run2_weight' \
--final_sparsity 0.9 --end_step 0.5 --ramping True

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --seed 3147 --run_name 'run3_weight' \
--final_sparsity 0.9 --end_step 0.5 --ramping True

# vgg sparse 95%
python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.95 --end_step 0.5 --ramping True

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --seed 128 --run_name 'run2_weight' \
--final_sparsity 0.95 --end_step 0.5 --ramping True

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --seed 3147 --run_name 'run3_weight' \
--final_sparsity 0.95 --end_step 0.5 --ramping True

# vgg sparse 98%
python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.98 --end_step 0.5 --ramping True

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --seed 128 --run_name 'run2_weight' \
--final_sparsity 0.98 --end_step 0.5 --ramping True

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --seed 3147 --run_name 'run3_weight' \
--final_sparsity 0.98 --end_step 0.5 --ramping True

# vgg sparse 98%
python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.1 --dataset cifar100 --run_name 'run1_weight_union' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle' \
--lr_schedule 'cyclic' --up_step 5000 --down_step 5000 --union_mask True

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.1 --dataset cifar100 --seed 128 --run_name 'run2_weight_union' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle' \
--lr_schedule 'cyclic' --up_step 5000 --down_step 5000 --union_mask True

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.1 --dataset cifar100 --seed 3147 --run_name 'run3_weight_union' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle' \
--lr_schedule 'cyclic' --up_step 5000 --down_step 5000 --union_mask True

# vgg sparse 98%
python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.1 --dataset cifar100 --run_name 'run1_weight' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle' \
--lr_schedule 'cyclic' --up_step 5000 --down_step 5000

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --seed 128 --run_name 'run2_weight' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle' \
--lr_schedule 'cyclic' --up_step 5000 --down_step 5000

python 'sparsity/main.py' --model vgg19_bn --model_type sparse --use_colab True \
--lr 0.05 --dataset cifar100 --seed 3147 --run_name 'run3_weight' \
--final_sparsity 0.9 --end_step 0.95 --ramping True --ramp_type 'half_cycle' \
--lr_schedule 'cyclic' --up_step 5000 --down_step 5000

