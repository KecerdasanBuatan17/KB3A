# In[16]: Dimana akan melakukan cross validation denga 5 split
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,dk_train_att,dk_train_label,cv=5)
scorerata2=scores.mean()
scorersd=scores.std()
# In[21]:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, dk_train_att, dk_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95
#% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),
scores.std() * 2))
# In[22]:
scorestree = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(),
scorestree.std() * 2))
# In[23]:
scoressvm = cross_val_score(clfsvm, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(),
scoressvm.std() * 2))
