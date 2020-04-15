# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:47:27 2020

@author: User
"""

import random
random.shuffle(imgs)
split_idx = int(0.8*len(imgs))
train = imgs[:split_idx]
test = imgs[split_idx:]