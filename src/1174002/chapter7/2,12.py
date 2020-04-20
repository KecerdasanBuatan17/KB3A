# In[12]: 5menit kali 10 epoch = 50 menit
# fungsi model titambahkan metod fit untuk mengetahui perhitungan dari train_input train_output
model.fit(train_input, train_output,
# dengan batch size 32 bit 
          batch_size=32,
          epochs=10,
          verbose=2,
          validation_split=0.2,
          callbacks=[tensorboard])

score = model.evaluate(test_input, test_output, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

