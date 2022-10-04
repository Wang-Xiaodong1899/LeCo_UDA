import os
import os.path as osp
import torch
import argparse
from module import network, loss
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import module.data_list as data_list
import math
from module.data_list import ImageList
from tqdm import trange
from utils import image_train,image_test
parser = argparse.ArgumentParser(description='show DEV loss')
parser.add_argument('--model_path', type=str, default='./model/CDAN.pt')
parser.add_argument('--mlp_path', type=str, default='./model/CDAN_mlp.pt')
parser.add_argument('--s', type=int, default=0)
parser.add_argument('--t', type=int, default=1)
parser.add_argument('--method', type=str, default='CDAN-E')
parser.add_argument('--trade_off',type=float,default='1')
parser.add_argument('--num_k',type=int,default=4)
parser.add_argument('--iterations',type=int,default=200)
parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
parser.add_argument('--net', type=str, default='resnet50',choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vgg11", "vgg13","vgg16", "vgg19", "vgg11bn", "vgg13bn", "vgg16bn", "vgg19bn", "alexnet"])
parser.add_argument('--da', type=str, default='uda', choices=['uda', 'ssda','uma','ssma'])
parser.add_argument('--dset', type=str, default='office', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
parser.add_argument('--worker', type=int, default=6, help="number of workers")
parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
parser.add_argument('--bottleneck', type=int, default=256)
parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
parser.add_argument('--output', type=str, default='origin')

args = parser.parse_args()
if args.dset == 'office-home':
    names = ['Art', 'Clipart', 'Product', 'Real']
    args.class_num = 65 
    args.root = '/data1/3/MM//data/office-home/images'
elif args.dset == 'office':
    names = ['amazon', 'dslr', 'webcam']
    args.class_num = 31
    args.root = '/data1/3/MM//data/office/domain_adaptation_images'
elif args.dset == 'DomainNet':
    names = ['clipart', 'painting', 'real','sketch']
    args.class_num = 126
    args.root = '/data1/3/MM//data/multi/domainnet'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
folder = './data/txt/'
args.s_dset_path = folder + args.dset + '/' + 'labeled_source_images_'+names[args.s] + '.txt'
args.t_dset_path = folder + args.dset + '/' + 'labeled_source_images_'+names[args.t] + '.txt'

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.t_dset_path).readlines()
    # dsize = len(txt_src)
    # tr_size = int(0.9*dsize)
    # src_train, src_val = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    #source train
    src_train = open(args.s_dset_path[:-4]+'_'+'tr'+'.txt').readlines()
    src_val = open(args.s_dset_path[:-4]+'_'+'va'+'.txt').readlines()
    dsets["source"] = ImageList(src_train,transform=image_train(),root = args.root)
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    #source vali
    dsets["vali"] = ImageList(src_val, transform=image_test(),root = args.root)
    dset_loaders["vali"] = DataLoader(dsets["vali"], batch_size=1, shuffle=False, num_workers=args.worker, drop_last=False)
    #target
    dsets["target"] = ImageList(txt_tar,transform=image_train(),root = args.root)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    #test
    dsets["test"] = ImageList(txt_test,transform=image_test(),root = args.root)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    len_tr = len(src_train)
    len_ts = len(txt_test)
    return dset_loaders, len_tr, len_ts

dset_loaders, len_tr, len_ts = data_load(args)
#networks load process
if args.net[0:3] == 'res':
    netG = network.ResBase(res_name=args.net).cuda()
elif args.net[0:3] == 'vgg':
    netG = network.VGGBase(vgg_name=args.net).cuda()

netB = network.feat_bootleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

#generate feature only
args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
args.savename = 'uda'+'_'+args.method+'_'+str(args.trade_off)
args.modelpath = args.output_dir_src + '/'+'target_G_'+args.savename+'.pt'   
netG.load_state_dict(torch.load(args.modelpath))
args.modelpath = args.output_dir_src + '/'+'target_B_'+args.savename+'.pt'   
netB.load_state_dict(torch.load(args.modelpath))
args.modelpath = args.output_dir_src + '/'+'target_C_'+args.savename+'.pt'   
netC.load_state_dict(torch.load(args.modelpath))

#attention!!!

def dev_loss_new(args):
    mlp = network.MLPClassifier(args.bottleneck).cuda()
    optimizer_mlp = optim.Adam(mlp.parameters(), lr=0.0001, weight_decay=0.0005)
    netG.eval()
    netB.eval()
    netC.eval()
    mlp.train()
    ##train mlp network
    for epoch in trange(200):
        #source
        try:
            inputs_s, label_s = iter_sour.next()
        except:
            iter_sour = iter(dset_loaders["source"])
            inputs_s, label_s = iter_sour.next()
        #target
        try:
            inputs_t, label_t = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_t, label_t = iter_target.next()
        
        inputs_s,label_s,inputs_t,label_t = inputs_s.cuda(),label_s.cuda(),inputs_t.cuda(),label_t.cuda()
        inputs_ = torch.cat((inputs_s,inputs_t),dim=0)
        feats_ = netB(netG(inputs_))
        mlp_loss = loss.mlp_loss(feats_,mlp)
        optimizer_mlp.zero_grad()
        mlp_loss.backward()
        optimizer_mlp.step()
    start_test = True
    mlp=mlp.eval()
    print(len(dset_loaders["vali"]))
    iter_vali = iter(dset_loaders["vali"])
    with torch.no_grad():
        for i in range(len(dset_loaders['vali'])):
            inputs_v, label_v = iter_vali.next()
            inputs_v,label_v = inputs_v.cuda(), label_v.cuda()
            feat_v = netB(netG(inputs_v))
            vali_outputs = netC(feat_v)
            vali_dlabel = mlp(feat_v)
            if start_test:
                all_dlabel = vali_dlabel
                _,predict = torch.max(vali_outputs,1)
                # print(predict,label_v)
                c_loss = np.array([int(predict!=label_v)])
                vali_loss = torch.from_numpy(c_loss)
                start_test = False
            else:
                all_dlabel = torch.cat((all_dlabel, vali_dlabel), 0)
                _,predict = torch.max(vali_outputs,1)
                # print(predict,label_v)
                c_loss = np.array([int(predict!=label_v)])
                c_loss = torch.from_numpy(c_loss)
                vali_loss = torch.cat((vali_loss, c_loss), 0)
    # print(all_dlabel)
    all_output = all_dlabel.cpu().detach().numpy()
    all_output = np.int64(all_output>=0.5)
    # print(all_output)
    # print(all_output.shape)
    # print(vali_loss)
    cl_acc = (np.squeeze(all_output)==np.zeros(all_output.shape[0])).sum()/all_output.shape[0]
    print('domain clf acc:',cl_acc)
    vali_loss = torch.unsqueeze(vali_loss, 1).float()
    print('origin vali_loss:', vali_loss.mean().item())
    # print('vali-sz:',vali_loss.size())
    # print('dlabel-sz:', all_dlabel.size())
    adlabel = torch.ones_like(all_dlabel) - all_dlabel
    wf = (len_tr / len_ts) * adlabel / all_dlabel  # modify
    new_weight = adlabel
    new_weight = new_weight.cuda()
    #print('wf sz:', wf.size())
    vali_loss = vali_loss.cuda()
    new_loss = new_weight * vali_loss
    new_loss = new_loss.mean()*(len_tr+len_ts)/len_ts
    print('my new_loss:', new_loss.item())
    wf = wf.cuda()
    weight_loss = wf * vali_loss
    # from IPython import embed;embed();exit();
    wf = wf.cpu().detach().numpy()
    weight_loss = weight_loss.cpu().detach().numpy()
    print('origin dev_loss:',weight_loss.mean())
    #cov = np.cov(np.concatenate((weight_loss, wf), axis=1), rowvar=False)[0][1]
    #var_w = np.var(wf, ddof=1)
    #eta = - cov / var_w
    # print('res:', np.mean(weight_loss), np.mean(wf), eta)
    #dev_l = np.mean(weight_loss) + eta * np.mean(wf) - eta
    return vali_loss.mean().item(),weight_loss.mean(), new_loss.item()

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
            outputs = netC(netB(netG(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

if __name__ == "__main__":
    dev_sum = 0.0
    Acc=[]
    TL=[]
    SL=[]
    DL=[]
    OL=[]
    # test(config)
    # dev_loss(config)
    acc_,_ = cal_acc(dset_loaders['test'],netG, netB, netC)
    print('accuracy:',acc_)
    for i in range(10):
       loss_data = dev_loss_new(args)
       SL.append(loss_data[0])
       DL.append(loss_data[1])
       OL.append(loss_data[2])
    mean, std = np.mean(SL), np.std(SL, ddof=1)
    print('Source loss:',mean,std)
    mean, std = np.mean(DL), np.std(DL, ddof=1)
    print('Dev loss:',mean,std)
    mean, std = np.mean(OL), np.std(OL, ddof=1)
    print('Our loss:',mean,std)
    print('accuracy:',acc_)
    #test(config)
    #test_gaussian(config)
