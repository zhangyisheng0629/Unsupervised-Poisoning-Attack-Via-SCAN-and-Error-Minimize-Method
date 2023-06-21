
# Labeling with clustering models (Gansbeke W etal.), generating noises with error-minimizing (Huang etal.) method.

Add experiments for label-agnostic dataset .  
  Supply AgnosticCIFAR10Folder etc. classes and some robust Loss functions for more robust training.
## Samplewise noise for UEs on agnostic-cifar10
Set the argument --seed , it will automatically genarates an experiment folder.  
The args --train_data_path, --test_data_path should be set to your own path.
```
--seed 4
--version resnet18
--exp_name result/agnostic_cifar10/min-min/samplewise/
--config_path configs/cifar10
--train_batch_size 512
--eval_batch_size 512
--num_of_workers 0
--train_data_type AgnosticCIFAR10Folder
--train_data_path C:\Users\zhangyisheng\Desktop\My-Unsupervised-Classification-master\datasets\agnostic-label-cifar-10-clean
--test_data_type CIFAR10
--test_data_path C:\Users\zhangyisheng\Desktop\My-Unsupervised-Classification-master\datasets\cifar-10
--universal_stop_error 0.01
--train_step 10
--attack_type min-min
--perturb_type samplewise
--noise_shape 50000 3 32 32
--epsilon 16
--num_steps 20
--step_size 0.8
```
Remenber set the arg --seed and the arg --perturb_tensor_filepath simultaneously.
## Classwise noise for UEs on agnostic-cifar10
```
--seed 4
--version resnet18
--exp_name result/agnostic_cifar10/min-min/samplewise/
--config_path configs/cifar10
--train_data_type PoisonAgnosticCIFAR10Folder
--poison_rate 1.0
--perturb_type samplewise
--perturb_tensor_filepath result/agnostic_cifar10/min-min/samplewise\resnet18_seed4/perturbation.pt
--train
--num_of_workers 0
--train_data_path C:\Users\zhangyisheng\Desktop\My-Unsupervised-Classification-master\datasets\agnostic-label-cifar-10-clean
--test_data_path C:\Users\zhangyisheng\Desktop\My-Unsupervised-Classification-master\datasets\cifar-10
```


# Aknowledgement
## Unlearnable Examples

Code for ICLR2021 Spotlight Paper ["Unlearnable Examples: Making Personal Data Unexploitable "](https://openreview.net/forum?id=iAmZUo0DxC0) by Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, Yisen Wang.

### Experiments in the paper.
Check scripts folder for *.sh for each corresponding experiments.

### Sample-wise noise for unlearnable example on CIFAR-10
###### Generate noise for unlearnable examples
```console
python3 perturbation.py --config_path             configs/cifar10                \
                        --exp_name                path/to/your/experiment/folder \
                        --version                 resnet18                       \
                        --train_data_type         CIFAR10                       \
                        --noise_shape             50000 3 32 32                  \
                        --epsilon                 8                              \
                        --num_steps               20                             \
                        --step_size               0.8                            \
                        --attack_type             min-min                        \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.01
```

