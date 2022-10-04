import random
seed =2021
random.seed(seed)
data = open('validation_list.txt').readlines()
class_idx = {}
for i in range(12):
    class_idx[i] = []
for i in data:
    class_idx[int(i.split()[-1])].append(i)

select=[]
rest=[]
for i in range(12):
    select_ = random.sample(class_idx[i],3)
    select.extend(select_)
    for se in select_:
        class_idx[i].remove(se)
    rest.extend(class_idx[i])
f = open('validation_labeled.txt','w')
for i in select:
    f.write(i)
f.close()
f = open('validation_unlabeled.txt','w')
for i in rest:
    f.write(i)
f.close()
