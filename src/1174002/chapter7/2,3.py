# In[3]:shuffle the data, split into 80% train, 20% test
# mengimport library random 
import random
# melakukan random pada vungsi imgs
random.shuffle(imgs)
# membuat variabel split_idx dengan nilai integer 80 persen dikali dari pengembalian jumlah dari variabel imgs
split_idx = int(0.8*len(imgs))
# membuat variabel train dengan isi lebih besar split idx
train = imgs[:split_idx]
# membuat variabel test dengan isi lebih kecil split idx
test = imgs[split_idx:]
