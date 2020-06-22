# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:27:57 2020

@author: User
"""

loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))