
# Labeling with clustering models (Gansbeke W etal.), generating noises with error-minimizing (Huang etal.) method.
## The repository is based on the Code from ICLR2021 Spotlight Paper ["Unlearnable Examples: Making Personal Data Unexploitable "](https://github.com/HanxunH/Unlearnable-Examples) by Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, Yisen Wang.
&nbsp;aa  
&emsp;aa
&ensp;aa  
好的

Add experiments for label-agnostic dataset .  
  Supply AgnosticCIFAR10Folder etc. classes and   
  some robust Loss functions for more robust training.  
  SCELoss ["Symmetric Cross Entropy for Robust Learning with Noisy Labels"](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf) by Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jin-Feng Yi, James Bailey.  
  Dowmload the agnostic-label CIFAR10 dataset from [here](https://drive.google.com/file/d/1Hs6zrwVeIhtAZti2ezG3XRyto5xuF-Cs/view?usp=drive_link).
## Generate agnostic label cifar10 min-min samplewise noise
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

## Train on agnostic label cifar10 min-min samplewise ue
Remenber set the arg --seed and the arg --perturb_tensor_filepath simultaneously.

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

## Generate agnostic label cifar10 min-min classwise noise
```
--seed 4
--config_path configs/cifar10
--exp_name result/agnostic_cifar10/min-min/classwise
--version resnet18
--train_data_type AgnosticCIFAR10Folder
--noise_shape 10 3 32 32
--epsilon 16
--num_steps 1
--step_size 0.8
--attack_type min-min
--perturb_type classwise
--universal_train_target train_subset
--universal_stop_error 0.1 --use_subset
--num_of_workers 0
--train_data_path C:\Users\zhangyisheng\Desktop\My-Unsupervised-Classification-master\datasets\agnostic-label-cifar-10-clean
--test_data_path C:\Users\zhangyisheng\Desktop\My-Unsupervised-Classification-master\datasets\cifar-10
```

## Train on agnostic label cifar10 min-min classwise ue
```
--seed 4
--version resnet18
--exp_name result/agnostic_cifar10/min-min/classwise/
--config_path configs/cifar10
--train_data_type PoisonAgnosticCIFAR10Folder
--poison_rate 1.0
--perturb_type classwise
--perturb_tensor_filepath result/agnostic_cifar10/min-min/classwise/resnet18_seed4\perturbation.pt
--train
--num_of_workers 0
--train_data_path C:\Users\zhangyisheng\Desktop\My-Unsupervised-Classification-master\datasets\agnostic-label-cifar-10-clean
--test_data_path C:\Users\zhangyisheng\Desktop\My-Unsupervised-Classification-master\datasets\cifar-10
```
## Aknowledgement
### Unlearnable Examples
ICLR2021 Spotlight Paper ["Unlearnable Examples: Making Personal Data Unexploitable "](https://openreview.net/forum?id=iAmZUo0DxC0) by Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, Yisen Wang.  
### SCAN
ECCV2020 ["SCAN: Learning to Classify Images without Labels"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550273.pdf) by Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, Luc van Gool.
