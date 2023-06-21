
# Labeling with clustering models (Gansbeke W etal.), generating noises with error-minimizing (Huang etal.) method.

Add experiments for label-agnostic dataset 
## Samplewise noise for UEs on agnostic-cifar10

# Aknowledgement
## Unlearnable Examples

Code for ICLR2021 Spotlight Paper ["Unlearnable Examples: Making Personal Data Unexploitable "](https://openreview.net/forum?id=iAmZUo0DxC0) by Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, Yisen Wang.

### Experiments in the paper.
Check scripts folder for *.sh for each corresponding experiments.
```
adasda
```
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

