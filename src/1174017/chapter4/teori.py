# Vektorisasi Data
# In[]:
import pandas as pd
d = pd.read_csv("Youtube01-Psy.csv")
# In[]:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
# In[]:
dvec = vectorizer.fit_transform(d['CONTENT'])
dvec
# In[]:
daptarkata = vectorizer.get_feature_names()
# In[]:
dshuf = d.sample(frac=1)
# In[]:
d_train = dshuf[:300]
d_test = dshuf[300:]
# In[]:
d_train_att = vectorizer.fit_transform(d_train['CONTENT'])
d_train_att
# In[]:
d_test_att = vectorizer.transform(d_test['CONTENT'])
d_test_att
# In[]:
d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']

# Klasifikasi Dengan Random Forest
# In[]:
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80)
# In[]:
clf.fit(d_train_att, d_train_label)
# In[]:
clf.predict(d_test_att, d_test_label)
# In[]:
clf.score(d_test_att, d_test_label)

# Confusion Matrix
# In[]:
from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(d_test_att)
cm = confusion_matrix(d_test_label, pred_labels)

# Pengecekan Cross Validation
# In[]:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)

skorrata2 = scores.mean()
skoresd = scores.std()