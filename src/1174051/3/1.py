import pandas as kue
nama_kue = {'Nama Kue' : ['Cuhcur', 'Putri Noong', 'Bugis', 'Papais', 'Ali-Ali'],
'Harga Satuan' : [2000,5000,1500,2500,1000], 'Terbilang' : ['Dua Ribu Rupiah', 'Lima Ribu Rupiah',
'Seribu Lima Ratus Rupiah', 'Dua Ribu Limaratus Rupiah', 'Seribu Rupiah']}
Data_kue = kue.DataFrame(nama_kue)
print(Data_kue)