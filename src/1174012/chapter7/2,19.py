# In[19]:restore the class name to integer encoder
# membuat variabel label encoder ke 2 dengan isian fungsi label encoder.
label_encoder2 = LabelEncoder()
# menambahkan method classess dengan data classess yang di eksport tadi
label_encoder2.classes_ = np.load('classes.npy')
# membuat fumgsi predict dengan path img
def predict(img_path):
    # membuat variabel newimg dengam membuay immage menjadi array dan membuka data berdasarkan img path
    newimg = keras.preprocessing.image.img_to_array(pil_image.open(img_path))
    # membagi data yang terdapat pada variabel newimg sebanyak 255
    newimg /= 255.0

    # do the prediction
    # membuat variabel predivtion dengan isian variabel model2 menggunakan fungsi predic dengan syarat variabel newimg dengan data reshape
    prediction = model2.predict(newimg.reshape(1, 32, 32, 3))

    # figure out which output neuron had the highest score, and reverse the one-hot encoding
    # membuat variabel inverted  denagan label encoder2 dan  menggunakan argmax untuk mencari skor luaran tertinggi
    inverted = label_encoder2.inverse_transform([np.argmax(prediction)])
    # mencetak prediksi gambar dan confidence dari gambar.
    print("Prediction: %s, confidence: %.2f" % (inverted[0], np.max(prediction)))

