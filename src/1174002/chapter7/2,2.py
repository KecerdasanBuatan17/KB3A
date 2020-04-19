# In[2]:load all images (as numpy arrays) and save their classes
#membuat variabel imgs dengan variabel kosong
imgs = []
#membuat variabel classes dengan variabel kosong
classes = []
#membuaka file hasy-data-labels.csv yang berada di folede HASYv2 yang di inisialisasi menjadi csvfile
with open('HASYv2/hasy-data-labels.csv') as csvfile:
    #membuat variabel csvreader yang berisi method csv.reader yang membaca variabel csvfile
    csvreader = csv.reader(csvfile)
    # membuat variabel i dengan isi 0
    i = 0
    # membuat looping pada variabel csvreader
    for row in csvreader:
        # dengan ketentuan jika i lebihkecil daripada o
        if i > 0:
            # dibuat variabel img dengan isi keras untuk aktivasi neural network fungsi yang membaca data yang berada dalam folder HASYv2 dengan input nilai -1.0 dan 1.0
            img = keras.preprocessing.image.img_to_array(pil_image.open("HASYv2/" + row[0]))
            # neuron activation functions behave best when input values are between 0.0 and 1.0 (or -1.0 and 1.0),
            # so we rescale each pixel value to be in the range 0.0 to 1.0 instead of 0-255
            #membagi data yang ada pada fungsi img sebanyak 255.0
            img /= 255.0
            # menambah nilai baru pada imgs pada row ke 1 2 dan dilanjutkan dengan variabel img
            imgs.append((row[0], row[2], img))
            # menambahkan nilai pada row ke 2 pada variabel classes
            classes.append(row[2])
            # penambahan nilai satu pada variabel i
        i += 1
