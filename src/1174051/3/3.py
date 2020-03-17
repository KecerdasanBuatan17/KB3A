import matplotlib.pyplot as plt

kelas_ti3 = ['Kelas A', 'Kelas B', 'Kelas C']
jumlah_mhs3 = [18, 14, 25]

kelas_ti2 = ['Kelas A', 'Kelas B', 'Kelas C']
jumlah_mhs2 = [22, 20, 24]

plt.figure(1, figsize=(9,3))
plt.subplot(131)
plt.bar(kelas_ti3, jumlah_mhs3)
plt.figure(2, figsize=(9,3))
plt.subplot(132)
plt.bar(kelas_ti2, jumlah_mhs2)
plt.suptitle('Categorical Plotting')
plt.show()
