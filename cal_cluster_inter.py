from turtle import color
from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rc('font',family='Times New Roman')

np.random.seed(2020)
random.seed(2020)
idxs = random.sample(range(55388),10000)
print(idxs[:10])
def cal_inter_std(method):
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
    centers=[]
    for i in range(12):
        cluster[i]=np.array(cluster[i])
        centers.append(cluster[i].mean(axis=0))
    centers = np.array(centers)
    print(centers.shape)    ##(12, 12)
    std = centers.std(axis=0)
    print(std.shape)    ##(12, 12)
    std = std.mean()
    return std
stds = []
stds.append(cal_inter_std('uda_CDAN_50_1_1_new'))
stds.append(cal_inter_std('uda_CDAN_L2_2.0_1_0.0_new'))
stds.append(cal_inter_std('uda_MCC_0.0_1_1_new'))
stds.append(cal_inter_std('uda_MCC_LECO_new_3.0_1_1_new'))
y = [0.3,0.6,0.9,1.2]
fig, ax = plt.subplots(figsize=(9,6))
colors = ['b','orange','g','r']
for i in range(4):
    plt.barh(y[i], stds[i], height=0.2, edgecolor='k',color=colors[i])
for i in range(len(stds)):
    plt.text(stds[i]+0.01,y[i],'{:.4f}'.format(stds[i]),fontsize=22)
# plt.title('INTER-class variance',fontsize=22)
plt.yticks(y,['CDAN','CDAN\n+LECO', 'MCC','MCC\n+LECO'],fontsize=22)
plt.xticks(fontsize=22)
plt.xlim((0,0.3))
for key, spine in ax.spines.items():
    # 'left', 'right', 'bottom', 'top'
    if key == 'right':
        spine.set_visible(False)

# plt.show()
plt.savefig('{}.png'.format('movti-b'),dpi=300)
