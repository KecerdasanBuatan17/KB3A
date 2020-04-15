# In[15]:join train and test data so we train the network on all data we have available to us
# melakukan join numpy menggunakan data train_input test_input
model.fit(np.concatenate((train_input, test_input)),
          # kelanjutan data yang di gunakan pada join train_output test_output
          np.concatenate((train_output, test_output)),
          #menggunakan ukuran 32 bit dan epoch 10 
          batch_size=32, epochs=10, verbose=2)

