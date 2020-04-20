# In[4]:
# mengimport librari numpy dengan inisial np
import numpy as np
# membuat variabel train input dengan np method asarray yang mana membuat array dengan isi row 2 dari data train
train_input = np.asarray(list(map(lambda row: row[2], train)))
# membuat test input input dengan np method asarray yang mana membuat array dengan isi row 2 dari data test
test_input = np.asarray(list(map(lambda row: row[2], test)))
# membuat variabel train_output dengan np method asarray yang mana membuat array dengan isi row 1 dari data train
train_output = np.asarray(list(map(lambda row: row[1], train)))
# membuat variabel test_output dengan np method asarray yang mana membuat array dengan isi row 1 dari data test
test_output = np.asarray(list(map(lambda row: row[1], test)))
