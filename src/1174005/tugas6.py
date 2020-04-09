# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:14:37 2020

@author: ONIWALDUS
"""
# In[1]: Soal Nomor 1

import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()
##Kode di atas menjelaskan isi data GTZAN. Ini adalah kumpulan data yang berisi 10 genre lagu dengan masing-masing genre memiliki
# 100 lagu yang akan kami lakukan proses MFCC dan juga freesound yang hanya berisi konten lagu, jika GTZAN memiliki beberapa genre 
#jika freesound hanya untuk 1 lagu dan disini kita membuat fungsi untuk membaca file audio dan outputnya sebagai plot.
# In[2]: Soal Nomor 2
display_mfcc('test1.wav')
# In[2]: Soal Nomor 2
display_mfcc('test2.wav')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/disco/disco.00069.au')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/blues/blues.00069.au')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/classical/classical.00069.au')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/country/country.00069.au')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/hiphop/hiphop.00069.au')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/jazz/jazz.00069.au')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/pop/pop.00069.au')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/reggae/reggae.00069.au')
# In[2]: Soal Nomor 2
display_mfcc('dataset/genres/rock/rock.00069.au')
#Kode di atas akan menampilkan hasil dari proses mfcc 
# yang sudah dibuat fungsi pada soal 1, yaitu display mfcc() dan akan menampilkan plot dari pembacaan file audio. 

# In[3]: Soal Nomor 3

def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]
##Baris pertama itu untuk membuat fungsi extract\_features\_song(f). Pada baris kedua itu akan me-load data 
#inputan dengan menggunakan librosa. Lalu selanjutnya untuk membuat sebuah fitur untuk mfcc dari y atau parameter inputan. 
#Lalu akan me-return menjadi array dan akan mengambil 25000 data saja dari hasil vektorisasi dalam 1 lagu. 


# In[4]: Soal Nomor 4

def generate_features_and_labels():
    all_features = []
    all_labels = []

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for genre in genres:
        sound_files = glob.glob('dataset/genres/'+genre+'/*.au')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding cth blues : 1000000000 classic 0100000000
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)#ke integer
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))#ke one hot
    return np.stack(all_features), onehot_labels
#Kode di atas dapat digunakan untuk melakukan fungsi yang sebelumnya telah kita lakukan. 
#Kemudian di bagian genre yang disesuaikan dengan dataset nama folder. Untuk baris berikutnya akan mengulang genre folder dengan ekstensi .au.# Maka itu akan memanggil fungsi ekstrak lagu. Setiap file dalam folder itu akan diekstraksi menjadi vektor dan akan ditambahkan ke fitur.
# Dan fungsi yang ditambahkan adalah untuk menumpuk file yang telah di-vektor-kan. Hasil kode tidak menampilkan output. 

# In[5]: Soal Nomor 5

features, labels = generate_features_and_labels()
# In[5]: Soal Nomor 5
print(np.shape(features))
print(np.shape(labels))

# Kode diatas berfungsi untuk melakukan load variabel features dan labels. Mengapa memakan waktu yang lama ?
# Karena mesin akan melakukan vektorisasi terhadap semua file yang berada pada setiap foldernya,
# di sini terdapat 10 folder dengan masing-masing folder terdiri atas 100 buah lagu, 
#setiap lagu tersebut akan dilakukan vektorisasi atau ekstraksi data menggunakan mfcc.

# In[6]: Soal Nomor 6
training_split = 0.8
# In[6]: Soal Nomor 6
alldata = np.column_stack((features, labels))
# In[6]: Soal Nomor 6
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]
# In[6]: Soal Nomor 6
print(np.shape(train))
print(np.shape(test))
# In[6]: Soal Nomor 6
train_input = train[:,:-10]
train_labels = train[:,-10:]
# In[6]: Soal Nomor 6
test_input = test[:,:-10]
test_labels = test[:,-10:]
# In[6]: Soal Nomor 6
print(np.shape(train_input))
print(np.shape(train_labels))
#Kode diatas berfungsi untuk melakukan training split 80\%. Karena supaya mesin dapat terus belajar tentang data baru,
# jadi ketika prediksi dibuat tentang data yang terlatih itu bisa mendapatkan persentase yang cukup bagus.

# In[7]: Soal Nomor 7
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])
    #fungsi Sequential() ialah Sebuah model untuk menentukan izin pada setiap neuron, 
    #di sini adalah 100 dense yang merupakan 100 neuron pertama dari data pelatihan.
    # Fungsi dari relay itu sendiri adalah untuk mengaktifkan neuron atau input yang memiliki nilai maksimum. 
    #Sedangkan untuk dense 10 itu adalah output dari hasil neuron yang telah berhasil diaktifkan, untuk dense 10 diaktifkan menggunakan softmax.
    
# In[8]: Soal Nomor 8
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
#Model Compile di perjelas dengan gambar dibawah, Hasil output pada kode tersebut seperti gambar  menjelaskan bahwa dense pertama 
# itu memiliki 100 neurons dengan parameter sekitar 2 juta lebih dengan aktviasi 100, 
#jadi untuk setiap neurons memiliki masing-masing 1 aktivasi. Sama halnya seperti dense 2 memiliki jumlah neurons 
# sebanyak 10 dengan parameter 1010 dan jumlah aktivasinya 10 untuk setiap neurons tersebut 
#dan total parameternya sekitar 2.5 juta data yang akan dilatih pada mesin tersebut.
	
# In[9]: Soal Nomor 9
model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)
#Kode tersebut berfungsi untuk melatih mesin dengan data training input dan training label. Epochs ini merupakan iterasi atau pengulangan berapa kali data tersebut akan dilakukan. Batch\_size ini adalah jumlah file yang akan dilakukan pelatihan pada setiap 1 kali pengulangan. 
#Sedangkan validation\_split itu untuk menentukan presentase dari cross validation atau k-fold sebanyak 20\% dari masing-masing data pengulangan.
# In[10]: Soal Nomor 10
loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
# In[10]: Soal Nomor 10
print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))
#Fungsi evaluate atau evaluasi ini ialah untuk menguji data pengujian setiap file. Di sini ada prediksi yang hilang,
# artinya mesin memprediksi data, sedangkan untuk keseluruhan perjanjian sekitar 55\%.

# In[11]: Soal Nomor 11
model.predict(test_input[:1])
#Fungsi Predict ialah untuk menghasilkan suatu nilai yang sudah di prediksi dari data training sebelumnya. 
#Gambar dibawah ini menjelaskan file yang di jalankan tersebut termasuk ke dalam genre apa, 
#hasilnya bisa dilihat pada gambar tersebut presentase yang paling besar yakni genre rock. 
#Maka lagu tersebut termasuk ke dalam genre rock dengan perbandingan presentase hasil prediksi.