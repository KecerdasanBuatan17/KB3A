# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:28:54 2020

@author: Arjun
"""

import os
import eyed3

root_folder = 'D:/KULIAH/DATA TINGKAT 3/SEMESTER 6/AI/UTS/DATASETLAGU'

files = os.listdir(root_folder)
if not files[1].endswith('.wav'):
    pass

for file_name in files:
    
    abs_location = '%s/%s' % (root_folder, file_name)
    
    song_info = eyed3.load(abs_location)
    if song_info is None:
        print('Skipping %s' % abs_location)
        continue
    
    print(song_info.tag.artist)