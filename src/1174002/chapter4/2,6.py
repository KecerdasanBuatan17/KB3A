# In[15]: Membuat confusion Matrix dan menampilkannya
from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(dk_test_att)
cm = confusion_matrix(dk_test_label, pred_labels)
cm
