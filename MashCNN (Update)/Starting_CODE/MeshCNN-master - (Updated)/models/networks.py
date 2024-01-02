import torch
import torch.nn as nn
import torch.nn.functional as F
# from mesh_conv import MeshConv
# from mesh_pool import MeshPool
from models.layers.mesh_conv import MeshConv
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
import functools
from torch.optim import lr_scheduler
from torch.nn import init

def define_loss(opt):
    if opt.dataset_mode == 'classification':
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    return loss

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args

class NoNorm(nn.Module): #todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()
    def forward(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)

def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


class MeshConvNet(nn.Module):
    def __init__(self, input_nc, ncf, ninput_edges, nclasses, pool_res, fc_n, resblocks, norm_type, num_groups):
        super(MeshConvNet, self).__init__()
        self.k = [input_nc] + ncf
        self.res = [ninput_edges] + pool_res

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pools = nn.ModuleList()

        norm_layer = get_norm_layer(norm_type=norm_type, num_groups=num_groups)
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            self.convs.append(MResConv(ki, self.k[i + 1], resblocks))
            self.norms.append(norm_layer(**norm_args[i]))
            self.pools.append(MeshPool(self.res[i + 1]))

        # self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_pool = nn.AvgPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):
        for i in range(len(self.k) - 1):
            x = self.convs[i](x, mesh)
            x = F.relu(self.norms[i](x))
            x = self.pools[i](x, mesh)

        x = self.global_pool(x)
        x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class MResConv(nn.Module):
    def __init__(self, nf0, nf1, resblocks):
        super(MResConv, self).__init__()
        self.conv1 = MeshConv(nf0, nf1, mesh_norm=False)
        self.norm1 = nn.InstanceNorm2d(nf1)
        self.conv2 = MeshConv(nf1, nf1, mesh_norm=False)
        self.norm2 = nn.InstanceNorm2d(nf1)
        self.skip = MeshConv(nf0, nf1, mesh_norm=False) if nf0 != nf1 else None
        self.resblocks = resblocks

    def forward(self, x, mesh):
        for _ in range(self.resblocks):
            identity = x

            x = self.conv1(x, mesh)
            x = self.norm1(x)
            x = F.relu(x)

            x = self.conv2(x, mesh)
            x = self.norm2(x)

            if self.skip is not None:
                identity = self.skip(identity, mesh)

            x += identity
            x = F.relu(x)

        return x