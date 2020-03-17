from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)
clf.fit(df_train_att, df_train_label)
print(clf.predict(df_train_att.head()))
clf.score(df_test_att, df_test_label)


