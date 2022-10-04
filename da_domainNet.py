import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch.autograd import grad
import warnings
warnings.filterwarnings('ignore')
# from tensorboardX import SummaryWriter

from module import network, loss
from module.data_list import ImageList, ImageList_idx,ImageList_NK
from utils import image_test, image_train, op_copy, lr_scheduler, TransformFixMatch, setup_seed


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    print('source list:',args.s_dset_path)
    sour_tr = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.t_dset_path_unl).readlines()
    dsize = len(sour_tr)
    tr_size = int(0.9*dsize)
    # torch.manual_seed(20)
    print('data root: ', args.root)
    ###deshow here
    if args.dset=='visda':
        s_root = osp.join(args.root, 'train')
        t_root = osp.join(args.root, 'validation')
    elif args.dset=='clef':
        names=['i','p','c']
        s_root = osp.join(args.root, names[args.s])
        t_root = osp.join(args.root, names[args.t])
    else:
        s_root = args.root
        t_root = args.root
    dsets["source"] = ImageList_idx(sour_tr, transform=image_train(),root = s_root)
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)

    # dsets["target"] = ImageList_idx(txt_tar, transform=image_train(),root = t_root)
    dsets["target"] = ImageList_idx(txt_tar, transform=TransformFixMatch(),root = t_root)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test(),root = t_root)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netG, netB, netC, flag=False):
    netG.eval()
    netB.eval()
    netC.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feat = netG(inputs)
            outputs = netC(netB(feat))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    netG.train()
    netB.train()
    netC.train()
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100

def train_target(args):
    setup_seed(args.seed)
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netG = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netG = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    if args.base[:4]=='CDAN':
        adnet = network.AdversarialNetwork(args.bottleneck*args.class_num,1024).cuda()
    elif args.base[:4]=='DANN':
        adnet = network.AdversarialNetwork(args.bottleneck,1024).cuda()

    #set eval
    netG.eval()
    netB.eval()
    netC.eval()
    
    param_group_1 = []
    for k, v in netG.named_parameters():
        if args.lr_decay1 > 0:
            param_group_1 += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay1 > 0:
            param_group_1 += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    if args.base[:4]=='CDAN' or args.base[:4]=='DANN':
        adnet.eval()
        for k, v in adnet.named_parameters():
            if args.lr_decay1 > 0:
                param_group_1 += [{'params': v, 'lr': args.lr * args.lr_decay1}]
            else:
                v.requires_grad = False
    param_group_2 = []
    for k, v in netC.named_parameters():
        if args.lr_decay2 > 0:
            param_group_2 += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    optimizer = optim.SGD(param_group_1)
    optimizer = op_copy(optimizer)
    optimizer_f = optim.SGD(param_group_2)
    optimizer_f = op_copy(optimizer_f)


    print('----------------------------------------------------------------\n')
    # max_iter = args.max_epoch * len(dset_loaders["target_unl"])
    max_iter = args.max_it
    interval_iter = max_iter // args.interval
    iter_num = 0
    pbar = tqdm(total=max_iter)
    device = "cuda"
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        netG = netG.to(device=device)
        netB = netB.to(device=device)
        netC = netC.to(device=device)
        netG = nn.DataParallel(netG)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
    
    while iter_num < max_iter:
        #source
        try:
            inputs_s, label_s, _ = iter_sour.next()
        except:
            iter_sour = iter(dset_loaders["source"])
            inputs_s, label_s, _= iter_sour.next()
        #target
        try:
            (inputs_w, inputs_st), label_t, idx_t = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            (inputs_w, inputs_st), label_t, idx_t = iter_target.next()

        netG.train()
        netB.train()
        netC.train()
        if args.base[:4]=='CDAN' or args.base[:4]=='DANN':
            adnet.train()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_f, iter_num=iter_num, max_iter=max_iter)
        batch_size = inputs_s.shape[0]
        inputs_s,inputs_w, inputs_st,label_s,label_t = inputs_s.to(device),inputs_w.to(device), inputs_st.to(device), label_s.to(device),label_t.to(device)
        inputs = torch.cat((inputs_s, inputs_w, inputs_st)).to(device)
        labels = torch.cat((label_s, label_t)).to(device)
        feat = netG(inputs)
        feat_ = netB(feat)
        target_ = netC(feat_)
        target_source = target_[:batch_size]
        target_weak = target_[batch_size:2*batch_size]
        target_strong = target_[-batch_size:]

        ### standard classifier_loss
        classifier_loss = F.cross_entropy(target_source, label_s, reduction='mean')

        if args.base == 'DANN':
            transfer_loss = loss.DANN(feat_[:2*batch_size],adnet,max_iter)
        elif args.base=='CDAN':
            target_ST = torch.cat((target_source,target_weak))
            softmax_target_ = nn.Softmax(dim=1)(target_ST)
            entropy_ = loss.Entropy(softmax_target_)
            transfer_loss = loss.CDAN([feat_[:2*batch_size],softmax_target_],adnet, max_iter,entropy_,network.calc_coeff(iter_num,max_iter=max_iter),None)
        elif args.base == 'BNM':#get
            softmax_target_ = nn.Softmax(dim=1)(target_weak)
            _, s_tgt, _ = torch.svd(softmax_target_)
            transfer_loss = -torch.mean(s_tgt)
        elif args.base == 'MCC':
            target_t1_ = nn.Softmax(dim=1)(target_weak)
            target_t2_ = nn.Softmax(dim=1)(target_strong)
            transfer_loss = loss.MCC(target_weak,args)
        elif args.base == 'SourceOnly':
            transfer_loss = 0
        else:
            raise ValueError('Baseline method cannot be recognized.')

        if args.method == 'NC':
            target_t1_ = nn.Softmax(dim=1)(target_weak)
            target_t2_ = nn.Softmax(dim=1)(target_strong)
            smo_loss  = ((target_t1_-target_t2_)**2).mean()
            if (iter_num-1)<=args.warm_up:
                smo_loss = 0
        elif args.method == 'LECO':
            target_t1_ = nn.Softmax(dim=1)(target_weak)
            target_t2_ = nn.Softmax(dim=1)(target_strong)
            sample_cfs = loss.sampleCC(target_weak,args).squeeze()
            tau = sample_cfs.mean()
            mask = (sample_cfs<(tau)).detach()
            l2_loss  = ((target_t1_-target_t2_)**2).mean(dim=1)
            smo_loss  = torch.sum(mask.view(-1, 1) * l2_loss.view(-1,1))
            if (iter_num-1)<=args.warm_up:
                smo_loss = 0
        elif args.method == 'ENT':
            smo_loss = loss.Hentropy(target_weak)
        elif args.method == 'Blank':
            smo_loss = 0.
        else:
            raise ValueError('Method cannot be recognized.')
        
        loss_total = classifier_loss + args.trade_off*transfer_loss + args.lamda*smo_loss
        
        optimizer.zero_grad()
        optimizer_f.zero_grad()
        loss_total.backward()
        optimizer.step()
        optimizer_f.step()
        pbar.update(1)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netG.eval()
            netB.eval()
            netC.eval()
            if args.dset=='visda':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netG, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te = cal_acc(dset_loaders['test'], netG, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netG.train()
            netB.train()
            netC.train()
    pbar.close()
    if args.issave:
        torch.save(netG.state_dict(), osp.join(args.output_dir, "target_G_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def check_badcase(args):
    dset = data_load(args)
    if args.net[0:3] == 'res':
        netG = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netG = network.VGGBase(vgg_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    savename = args.da+'_'+args.method+'_warm_up='+str(args.warm_up)+'_lamda_reg='+str(args.lamda_reg)+'_lamda_en='+str(args.lamda_en)+'_mask_nums='+str(args.mask_nums)
    print(savename)
    savedir = osp.join(args.output_dir,savename)
    args.modelpath = args.output_dir + '/'+'target_G_'+savename+'.pt'
    netG.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/'+'target_B_'+savename+'.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/'+'target_C_'+savename+'.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    acc, acc_list, pseudo_list, all_label = cal_acc(dset['test'], netG, netB, netC, True)
    print('acc: ', acc)
    print(acc_list)
    t_pth_pseu = savedir + '_pred.txt'
    f_tgt = open(t_pth_pseu,'w')
    for idx,item in enumerate(pseudo_list):
        f_tgt.write(str(item)+' '+str(int(all_label[idx]))+'\n')
    f_tgt.close()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LECO for DomainNet')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=6, help="number of workers")
    parser.add_argument('--dset', type=str, default='com-dn', choices=['clef','visda', 'office', 'office-home', 'com-dn'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="alexnet, vgg16, resnet50, res101,resnet34")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='log')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'ssda','uma','ssma'])
    parser.add_argument('--issave', action='store_true',help='save model checkpoint or not')
    parser.add_argument('--temperature', type=float, default=2.5, metavar='T',help='temperature for MCC(default: 2.5)')

    parser.add_argument('--trade_off', default=1, type=float,help='trade off between classfication loss and transfer loss')
    parser.add_argument('--max_it', default=10000, type=int, help='max number of iterations')
    
    parser.add_argument('--base', type=str, default='MCC',help='baseline methods')
    parser.add_argument('--method', type=str, default='LECO',help='smoothness loss',choices=['Blank', 'NC', 'LECO', 'ENT'])

    parser.add_argument('--lamda', default=3, type=float,help='lamda for smoothness loss')
    parser.add_argument('--warm_up', default=3000, type=int,help='warm up iterations')

    args = parser.parse_args()
    print(args.net)
    print('---------------------------------------------------------')
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.class_num = 65
        args.root = '/home/21/DAlib/dsets/office-home/images'
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        args.root = '/home/21/DAlib/dsets/office'
    elif args.dset == 'com-dn':
        names = ['clipart', 'painting','real','sketch']
        args.class_num = 345
        args.root = '/home/21/DAlib/dsets/complete-domainnet'
    elif args.dset == 'visda':
        names = ['train', 'validation']
        args.class_num = 12
        args.root = '/dsets/visda'
    elif args.dset == 'clef':
        names = ['i', 'p', 'c']
        args.class_num = 12
        args.root = '/home/21/DAlib/dsets/clef'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print('gpu: ',os.environ["CUDA_VISIBLE_DEVICES"])
    print('seed: ', args.seed)
    cnt=0
    acc_total = []
    for s in range(len(names)):
        args.s = s
        for i in range(len(names)):
            if i== args.s:
                continue
            args.t = i
            print(f'da: {args.da}, baseline: {args.base}, method: {args.method}')
            print(f'task: {names[args.s]}->{names[args.t]}')
            args.folder = './data/txt/'
            if args.da == 'uda' or args.da == 'uma':
                args.s_dset_path = args.folder + args.dset + '/' + 'labeled_source_images_'+names[args.s] + '.txt'
                args.t_dset_path = args.folder + args.dset + '/' + 'labeled_source_images_'+names[args.t] + '.txt'
                args.t_dset_path_unl = args.folder + args.dset + '/' + 'labeled_source_images_'+names[args.t] +'.txt'
                if args.dset == 'clef':
                    args.s_dset_path = args.folder + args.dset + '/' +names[args.s] + '_list.txt'
                    args.t_dset_path = args.folder + args.dset + '/' + names[args.t] + '_list.txt'
                    args.t_dset_path_unl = args.folder + args.dset + '/' +names[args.t] +'_list.txt'
                elif args.dset=='visda':
                    args.s_dset_path = args.folder + args.dset + '/' +names[args.s] + '.txt'
                    args.t_dset_path = args.folder + args.dset + '/' + names[args.t] + '.txt'
                    args.t_dset_path_unl = args.folder + args.dset + '/' +names[args.t] +'.txt'

            # args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
            args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
            args.name = names[args.s][0].upper()+names[args.t][0].upper()

            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)
            args.savename = f'{args.base}_{args.method}_{args.lamda}_{args.seed}_'
            args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')

            args.out_file.write(print_args(args)+'\n')
            args.out_file.flush()
            train_target(args)
