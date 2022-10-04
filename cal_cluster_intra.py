from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
plt.rc('font',family='Times New Roman')

np.random.seed(2020)
random.seed(2020)
idxs = random.sample(range(55388),10000)
print(idxs[:10])
method = 'Ours'
def cal_intra_std(method):
    X = np.load('logits_{}.npy'.format(method))
    Y = np.load('labels_{}.npy'.format('MCC'))
    x = X[idxs]
    y = Y[idxs]
    cluster={}
    for i in range(12):
        cluster[i]=[]
    for id in idxs:
        x,y = X[id],Y[id]
        y=int(y)
        cluster[y].append(x)
    sum=0
    stds=[]
    for i in range(12):
        cluster[i]=np.array(cluster[i])
        std = cluster[i].std(axis=0).mean()
        stds.append(std)
    return stds
# std_ours = cal_std('Ours')
std_CDAN = cal_intra_std('uda_CDAN_50_1_1_new')
std_CDAN_L = cal_intra_std('uda_CDAN_L2_2.0_1_0.0_new')
std_MCC = cal_intra_std('uda_MCC_0.0_1_1_new')
std_MCC_L = cal_intra_std('uda_MCC_LECO_new_3.0_1_1_new')
std_CDAN = np.array(std_CDAN)
std_CDAN_L = np.array(std_CDAN_L)
std_MCC = np.array(std_MCC)
std_MCC_L = np.array(std_MCC_L)
means = []
means.append(std_CDAN.mean())
means.append(std_CDAN_L.mean())
means.append(std_MCC.mean())
means.append(std_MCC_L.mean())
print(means)
y = [0.3,0.6,0.9,1.2]
colors = ['b','orange','g','r']
fig, ax = plt.subplots(figsize=(9,6))
for i in range(4):
    plt.barh(y[i],means[i],height=0.2,edgecolor='k',color=colors[i])
for i in range(len(means)):
    plt.text(means[i]+0.005,y[i],'{:.4f}'.format(means[i]),fontsize=22)
# x = np.array(range(12))
# plt.bar(x=x-0.3,height=std_L2,width=0.3,color='b',label='L2')
# plt.bar(x=x,height=std_mcc,width=0.3,color='orange',label='MCC')
# plt.bar(x=x+0.3,height=std_dann,width=0.3,color='black',label='DANN')
# name = ['plane','bcycl','bus', 'car', 'horse', 'knife', 'mcycl', 'person', 'plant', 'sktbrd', 'train', 'truck']
# plt.xticks(range(12),name,rotation=45)
# plt.title('classification-response INTRA-class variation',fontsize=18)
plt.yticks(y,['CDAN','CDAN\n+LECO','MCC','MCC\n+LECO'],fontsize=22)
plt.xticks(fontsize=22)
# plt.show()
# plt.legend()
# plt.show()
# plt.ylim((0,0.2))
for key, spine in ax.spines.items():
    # 'left', 'right', 'bottom', 'top'
    if key == 'right':
        spine.set_visible(False)
plt.xlim((0,0.14))
plt.savefig('{}.png'.format('movti-a'),dpi=300)