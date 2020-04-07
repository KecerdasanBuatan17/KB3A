
print(np.shape(features))
print(np.shape(labels))
training_split = 0.8
# last column has genre, turn it into unique ids
alldata = np.column_stack((features, labels))
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]
print(np.shape(train))
print(np.shape(test))
train_input = train[:,:-10]
train_labels = train[:,-10:]
test_input = test[:,:-10]
test_labels = test[:,-10:]
print(np.shape(train_input))
print(np.shape(train_labels))