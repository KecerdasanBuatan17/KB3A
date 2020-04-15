# In[18]:load the pre-trained model and predict the math symbol for an arbitrary image;
# the code below could be placed in a separate file
# mengimpport librari keras model
import keras.models
# membuat variabel model2 untuk meload model yang telah di simpan tadi
model2 = keras.models.load_model("mathsymbols.model")
# mencetak hasil model2
print(model2.summary())

