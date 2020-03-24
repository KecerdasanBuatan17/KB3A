# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 07:46:59 2020

@author: dzihan
"""
# In[44]: Soal1
import pandas as pd

# Import the cars.csv data: cars
cars = pd.read_csv('cars.csv')

# Print out cars
print(cars)
# In[44]: Soal1
import numpy as np
a = np.array([1,2,3])
a
# In[44]: Soal1
import matplotlib.pyplot as plt
x = [2,4,6,7,9,13,19,26,29,31,36,40,48,51,57,67,69,71,78,88]
y = [54,72,43,2,8,98,109,5,35,28,48,83,94,84,73,11,464,75,200,54]
plt.scatter(x,y)
plt.show()