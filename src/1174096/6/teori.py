# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:38:56 2020

@author: Nico Sembiring
"""
# In[soal 7]:
import numpy as np
import keras.utils as ku
arr = [1,5,2,3,3,2,3,3,3,1,2,3,4,5,1,2,5,3,7,8,2,3,7,5,7,2]
print("Awalnya Gini")
print(arr)
#NP Unique akan melepas nilai yang sama
print("Jadinya gini kalau pake np.unique")
print(np.unique(arr))
print("Pakai To Categorical (From keras.utils)")
print(ku.to_categorical(np.unique(arr), num_classes=None))

# In[soal 8]:
target = 7
arr = [1,5,2,3,3,2,3,3,3,1,2,3,4,5,1,2,5,3,7,8,2,3,7,5,7,2]
for x in range(0, len(arr)):
if target == arr[x]:
    print("\nNilai "+str(target)+" ada")
    break