# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:21:58 2020

@author: User
"""

import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical