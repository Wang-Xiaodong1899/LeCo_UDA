import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch.autograd import grad


def Hentropy(outputs, lamda=1):
    out_t1 = F.softmax(outputs,dim=1)
    loss_ent = -lamda * torch.mean(out_t1 *
                                             (torch.log(out_t1 + 1e-5)))
    return loss_ent

def Entropy(input_,dd = None):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if dd!=None:
        entropy = torch.sum(entropy, dim=dd)
    else:
        entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net,max_iter, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    # print('softmax size:',softmax_output.size())
    # print('feat size:',feature.size())
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        # print('opt size: ',op_out.size())
        # print('max',max_iter)
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)),max_iter=max_iter)
        # print('view size: ',op_out.view(-1, softmax_output.size(1) * feature.size(1)).size())

    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)),max_iter=max_iter)
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

def DANN(features, ad_net,max_iter):
    ad_out = ad_net(features,max_iter)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


def sampleCC(outputs_target,args):
    outputs_target_temp = outputs_target / args.temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    sample_confusion = torch.zeros(args.batch_size,1).cuda()
    for idx in range(args.batch_size):
        cov_ma = target_softmax_out_temp[idx].view(1,-1).transpose(1,0).mm(target_softmax_out_temp[idx].view(1,-1))
        sample_confusion[idx] = torch.sum(cov_ma) - torch.trace(cov_ma)
    return sample_confusion

def MCC(outputs_target,args):
    outputs_target_temp = outputs_target / args.temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = args.batch_size * target_entropy_weight / torch.sum(target_entropy_weight)
    cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(target_softmax_out_temp)
    cov_matrix_t = target_softmax_out_temp.transpose(1,0).mm(target_softmax_out_temp)

    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / args.class_num
    return mcc_loss
