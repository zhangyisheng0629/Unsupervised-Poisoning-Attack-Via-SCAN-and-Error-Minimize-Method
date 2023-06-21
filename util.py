import logging
import os

import numpy as np
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def _patch_noise_extend_to_img(noise, image_size=[3, 32, 32], patch_location='center'):
    c, h, w = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((c, h, w), np.float32)
    x_len, y_len = noise.shape[1], noise.shape[2]

    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise('Invalid patch location')

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[:, x1: x2, y1: y2] = noise
    return mask


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = '\t' + key + '=' + value
        else:
            display += '\t' + str(key) + '=%.4f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1/batch_size))
    return res


def save_model(filename, epoch, model, optimizer, scheduler, save_best=False, **kwargs):
    # Torch Save State Dict
    state = {
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, filename + '.pth')
    filename += '_best.pth'
    if save_best:
        torch.save(state, filename)
    return


def load_model(filename, model, optimizer, scheduler, **kwargs):
    # Load Torch State Dict
    filename = filename + '.pth'
    checkpoints = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoints['model_state_dict'])
    if optimizer is not None and checkpoints['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    if scheduler is not None and checkpoints['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    return checkpoints


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_model(p, pretrain_path=None, transforms=None):
    # Get backbone
    if p['backbone'] == 'resnet18':
        if p['train_db_name'] in ['cifar-10', 'cifar-20']:
            from models.ClusteringCifarResNet import resnet18
            backbone = resnet18()

        elif p['train_db_name'] == 'stl-10':
            from models.resnet_stl import resnet18
            backbone = resnet18()

        # Add MNIST
        elif p['train_db_name'] == 'mnist':
            from models.resnet_mnist import resnet18
            backbone = resnet18()

        else:
            raise NotImplementedError

    elif p['backbone'] == 'resnet50':
        if 'imagenet' in p['train_db_name']:
            from models.resnet import resnet50
            backbone = resnet50()
    elif p['backbone'] == 'supervised_resnet18':
        if p['train_db_name'] in ['cifar-10', 'cifar-10-tight', 'cifar-10-pgdscanloss', 'cifar-20']:
            from supervised.supervised_models.resnet_cifar import resnet18
            backbone = resnet18()
        elif p['train_db_name'] == 'stl-10':
            from supervised.supervised_models.resnet_stl import resnet18
            backbone = resnet18()

        else:
            raise NotImplementedError

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    if p['setup'] in ['simclr', 'moco']:
        from models.ClusteringModel import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    elif p['setup'] in ['scan', 'selflabel']:
        from models.ClusteringModel import ClusteringModel
        if p['setup'] == 'selflabel':
            assert (p['num_heads'] == 1)
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'], transform=transforms)
    elif p['setup'] == 'supervised':
        from models.ClusteringModel import SupervisedModel
        model = SupervisedModel(backbone, transform=transforms)

    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')
        #
        if p['setup'] == 'simclr':
            model.load_state_dict(state, strict=False)

        elif p['setup'] == 'scan':  # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state, strict=False)
            assert (set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias',
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                    or set(missing[1]) == {
                        'contrastive_head.weight', 'contrastive_head.bias'})

        elif p['setup'] == 'selflabel':  # Weights are supposed to be transfered from scan
            # We only continue with the best head (pop all heads first, then copy back the best head)
            model_state = state['model']
            all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
            best_head_weight = model_state['cluster_head.%d.weight' % (state['head'])]
            best_head_bias = model_state['cluster_head.%d.bias' % (state['head'])]
            for k in all_heads:
                model_state.pop(k)

            model_state['cluster_head.0.weight'] = best_head_weight
            model_state['cluster_head.0.bias'] = best_head_bias
            missing = model.load_state_dict(model_state, strict=True)

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model