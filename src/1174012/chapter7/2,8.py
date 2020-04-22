# In[8]:convert train and test output to one-hot
# mengkonvert data train output  mengguanakn variabel label_encoder kedalam variabel train_output_int
train_output_int = label_encoder.transform(train_output)
# mengkonvert variabel train_output_int kedalam fungsi onehot_encoder 
train_output = onehot_encoder.transform(train_output_int.reshape(len(train_output_int), 1))
# mengkonvert data test_output mengguanakn variabel label_encoder kedalam variabel test_output_int
test_output_int = label_encoder.transform(test_output)
# mengkonvert variabel test_output_int kedalam fungsi onehot_encoder 
test_output = onehot_encoder.transform(test_output_int.reshape(len(test_output_int), 1))
# membuat variabel num_classes dengan isi variabel label_encoder dan classess
num_classes = len(label_encoder.classes_)
# mencetak hasil dari nomer Class beruapa persen 
print("Number of classes: %d" % num_classes)

