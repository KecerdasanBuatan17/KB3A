# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:34:10 2020

@author: Sujadi
"""

loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))