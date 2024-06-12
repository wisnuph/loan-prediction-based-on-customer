# Loan Prediction Based on Customer Behavior

Final Project Bootcamp Data Science Rakamin Academy - Batch 42

|     | **PSM (Pejuang Sabtu Malam)**      |
| --- | ---------------------------------- |
| 1   | Wisnu Pri Hartono (Ketua Kelompok) |
| 2   | Muhammad Zulfarhan                 |
| 3   | Arman Lukman                       |
| 4   | Farki Mahbubi                      |
| 5   | Radithya Arif Pambudi              |
| 6   | Surya Praviarti                    |
| 7   | Raihan Damar                       |

## Dataset

Bersumber dari Kaggle Dataset dengan judul [Loan Prediction Based on Customer Behavior](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)

Dalam Dataset diperoleh Data Total nasabah yaitu 252.000 nasabah.

Terdiri dari 13 Column :

- **Id, Income, Age, Experience, Married/Single, House_Ownership, Car_Ownership, Profession, CITY, STATE, CURRENT_JOB_YRS, CURRENT_HOUSE_YRS, Risk_Flag**

## Insight Summary dari [Exploratory Data Analysis](https://colab.research.google.com/drive/1dkZTO5T_3kKiRdpLPdupE97ANdhv2t2L?usp=sharing)

Berdasarkan data yang diperoleh dapat disimpulkan bahwa tidak adanya nilai outliers dan distribusi data rata-rata cenderung normal kecuali pada kolom CURRENT_JOB_YEARS yang memiliki distribusi skewed ke arah kanan.

Dalam penentuan tingkat resiko gagal bayar (risk flag) yang dilakukan oleh nasabah, dapat diterapkan analisa berdasarkan:

- **group usia vs group income**

  Group usia dan group income sama-sama lebih dominan tidak beresiko risk flag dari pada yang beresiko risk flag.
  Jumlah nasabah yang tidak beresiko risk flag yang terbesar yaitu pada group usia dewasa dan group low income sebesar 25501. Sebaliknya yang terkecil adalah group usia remaja dan group low income sebesar 5969.
  Jumlah nasabah yang beresiko risk flag yang terbesar yaitu pada group usia dewasa dan group low income sebesar 3788. Sebaliknya yang terkecil adalah group usia remaja dan group medium income sebesar 811.

- **married status vs group income**

  Berdasarkan data diperoleh bahwa nasabah dengan status single dan rentang pendapatan High Income cenderung memiliki tingkat resiko cukup tinggi dalam gagal bayar pinjaman (memiliki risk flag yang terbesar) sebesar 9829.
  Sebaliknya untuk nasabah dengan status married dan rentang pendapatan High Income justru memiliki tingkat risk flag terendah sebesar 799.

- **group income vs house ownership**

  Group income dan group house ownership sama-sama lebih dominan tidak beresiko risk flag dari pada yang beresiko risk flag.
  Berdasarkan data diperoleh bahwa group rented yang memiliki resiko risk flag terbesar sekaligus tidak memiliki resiko risk flag yang terbesar.
  Kemudian pada group norent noown yang memiliki resiko risk flag terkecil sekaligus tidak memiliki resiko risk flag yang terkecil.
  Jumlah nasabah yang tidak beresiko risk flag yang terbesar yaitu pada group high income dan group rented sebesar 67836.
  Sebaliknya yang terkecil adalah group medium income dan group norent noown sebesar 1844.
  Jumlah nasabah yang beresiko risk flag yang terbesar yaitu pada group high income dan group rented sebesar 10062.
  Sebaliknya yang terkecil adalah group medium income dan group norent noown sebesar 193.

Selain itu, dapat juga diterapkan analisis dan visualisasi data berdasarkan:

- **usia vs status**

  Tingkat resiko berdasarkan usia dan status yang tertinggi yaitu pada usia Dewasa dengan status single sebesar 14.99% atau 9.584 customer beresiko.
  Berdasarkan data ini pemberi pinjaman dapat memberikan penambahan jaminan pada customer dengan usia Dewasa dan status Single untuk memperkecil kemungkinan kerugian perusahaan saat customer tersebut berisiko.

- **state**

  Dari 10 state dengan jumlah pelanggan terbanyak, tingkat resiko tertinggi terdapat di state Uttar Pradesh sebanyak 3.343 nasabah beresiko. Top 10 Jumlah nasabah beresiko di state Uttar Prades di dominasi kepemilikan rumah sewa dan tidak mempunyai mobil dengan Income > 2,000,000.
  Berdasarkan data tersebut, dengan melakukan Clusterisasi daerah yang beresiko tinggi dan mewajibkan DP tinggi agar cicilan customer rendah sehingga customer tidak berat saat membayar pinjaman.
  Dari segi bisnis perusahaan dapat menaikkan Bunga pinjaman agar modal yang dikeluarkan untuk pemberian pinjaman dapat kembali diawal.

- **harta kepemilikan dan income dengan jumlah user**

  Berdasarkan Income, Kepemilikan Rumah dan Mobil dapat dibuat sebuah penilaian harta kepemilikan(Net Worth Value) dari User. Berdasarkan Incomenya dapat diberikan nilai 1-3 dari Low, Medium dan High Income berdasarkan pembagian dari Q1 dan Q3 Income. Untuk Kepemilikan rumah diberikan nilai 3 untuk ‘owned’, 2 untuk ‘rented’ dan 1 untuk ‘noown_norent’ dan kemudian dijumlahkan. Untuk kepemilikan mobil digunakan 1 untuk ‘yes’ dan 0 untuk ‘no’. Berdasarkan grafik dari data diperoleh bahwa user dominan memiliki harta kekayaan baik itu dari Income, rumah ataupun mobil. User dengan value score 4 memiliki jumlah terbanyak dan juga memiliki risk yang besar.
  Dari data tersebut, pemberian kredit lebih baik diberikan kepada nasabah yang memiliki networth value yang bernilai 4 kebawah dapat dilakukan penambahan bunga atau peningkatan DP untuk mengurangi kerugian apabila user tersebut gagal bayar.

## Insight Summary dari [Data Pre-Processing](https://colab.research.google.com/drive/1S3XWj_aQnOsSFXGh6Ej67QZyJrtYzHMk?usp=sharing)

Kemudian tim kami melakukan data Pre-processing yakni dengan melakukan tahapan berikut ini:

- **Data Cleansing**

  Setelah dilakukan pengecekan terhadap data yang kosong atau null-value, ternyata tidak terdapat missing values. Kemudian juga tidak terdapat data yang bersifat duplikat.

  Saat dicek terhadap deskripsi data yang berfokus pada fitur numerik, kita menganalisa dari nilai mean dengan median setiap fitur tersebut. Ternyata setiap fitur memiliki nilai mean dan median yang tidak berbeda signifikan sehingga distribusi cenderung normal. Oleh karena itu tidak dilakukan feature transformation.

  Saat kita membuat boxplot dari setiap fitur numerik, tidak ditemukan adanya outlier data. Maka dari itu kita tidak melakukan handling outlier tersebut.

  Pada tahapan encoding, kami melakukan tahap tersebut pada fitur CAR_OWNERSHIP, MARRIED/SINGLE, dan HOUSE_OWNERSHIP yakni dengan mengubah fitur kategorikal menjadi numerik dengan memberikan angka yang berbeda bagi masing masing nilai unique.

  Kami mengecek dari count value dari label Risk_Flag, ternyata dihasilkan perbandingan pada label 0 dan 1 yaitu 88:12. Nilai proporsi ini kurang cocok pada kebutuhan pemodelan karena akan cenderung menghasilkan nilai label 0. Oleh karena itu kita melakukan handling imbalance data dengan teknik SMOTE agar proporsi label 0 dan 1 sama. Setelah dilakukan teknik SMOTE, diperoleh nilai label 0 dengan 1 yaitu sebanyak 221004.

- **Feature Engineering**

  Ketika dibuat heatmap sebagai korelasi antar fitur, ternyata tidak ada nilai yang mempunyai korelasi kuat antar setiap fitur sehingga tidak menyebabkan redundan data. Oleh karena itu tim kami memilih semua fitur numerik dalam kebutuhan model machine learning.

  Dari kolom Age, tim kami melakukan feature extract dengan membuat kolom baru Group_Age yaitu dengan kelompok usia (Berdasarkan Depkes RI tahun 2009):

  - Remaja -> 20 tahun hingga 25 tahun
  - Dewasa -> 26 tahun hingga 45 tahun
  - Lansia -> 46 tahun hingga 65 tahun
  - Manula -> lebih dari 65 tahun

    Dalam analisa kredit, pengambilan keputusan dapat diberikan dengan informasi terkait grup usia, dan juga bermanfaat pada pemasaran yang efektif.

  Dari kolom Income, dilakukan extract fitur dengan membuat Group_Income yakni menjadi 3 kategori sesuai Quartile, yang dihasilkan Low, Medium, dan High income. Hal ini dapat memberikan instrumen penting bagi pengambilan keputusan kredit dengan penilaian resiko kredit dan pengembangan strategi pemasaran bagi bank.

  Kami mencoba juga membuat kolom prod_yrs_left dimana sebagai tolak ukur sisa produktif dari seseorang yang diambil dari batas pensiun = 64 tahun (berdasarkan UU 13 Tahun 2003,Bab 1 Pasal 1 Ayat 2) dikurangi dengan Umur seseorang. Hal ini tentunya dapat bermanfaat bagi lembaga keuangan untuk kebutuhan evaluasi potensi penghasilan, stabilitas kerja, dan lainnya yang dapat dijadikan sebagai bahan pertimbangan untuk keputusan peminjaman kredit.

  Tim kami menentukan juga terkait fitur tambahan yang dapat digunakan dalam peningkatan performa model yakni antara lain:

  - Jumlah Tanggungan
  - Total Hutang
  - Limit Pinjaman
  - Status BI Checking
  - Jumlah Aset lainnya
  - Business Ownership
  - Income lainnya

## Insight Summary dari [Modelling and Evaluation](https://colab.research.google.com/drive/1S3XWj_aQnOsSFXGh6Ej67QZyJrtYzHMk?usp=sharing)

Pada tahap ini kami menggunakan tiga algoritma selama proses modelling yang diantaranya:

- **Logistic Regression**

![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/e4c6ac42-0bf2-46fb-87bc-d5795cdaf616)
![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/6915cab2-db71-41c7-8e04-7612dcc61dd3)
![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/b822fb41-7ac3-4891-afaa-2d144e8a24d4)

Dari hasil pemodelan dengan algoritma Logistic Regression secara default, mendapatkan nilai seperti pada tabel diatas. Dari hasil tersebut didapatkan nilai metrics yang masih terbilang rendah akan tetapi dari nilai test dan train yang didapatkan model sangat fit.

Berdasarkan nilai recallnya (50%) apabila kita memiliki 100 orang nasabah yang resiko gagal bayar maka model logistic regression hanya dapat menangkap 50 orang diantaranya.

- **K-Nearest Neighbor(KNN)**

![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/38efcba6-42c0-42c3-a595-4249ed30606f)
![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/2d1cd098-8fd7-49e2-8ca9-95030e18787d)
![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/34977606-5ed0-461b-bfa8-455bd319c41e)

Dari hasil pemodelan dengan algoritma K-Nearest Neighbour secara default, mendapatkan nilai seperti pada tabel diatas. Dari hasil tersebut didapatkan nilai metrics yang lebih baik dari hasil Logistic Regression. Model tersebut memiliki overfit sekitar 2% untuk metrics-metricsnya

Berdasarkan nilai recallnya (91%) apabila kita memiliki 100 orang nasabah yang resiko gagal bayar maka model KNN hanya dapat menangkap 91 orang diantaranya.

- **Random Forest**

![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/a7dfeeb7-80d2-4c0e-8e89-637c013c9b14)
![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/ef9c47ce-a33c-4d03-9b0e-340b047e7bb9)
![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/7004075f-ab79-4b73-b724-553b2f2d6fa5)

Dari hasil pemodelan dengan algoritma Random Forest secara default, mendapatkan nilai seperti pada tabel diatas. Dari hasil tersebut didapatkan nilai metrics yang lebih baik lagi dari hasil KNN. Model tersebut memiliki overfit sekitar 3%-4% untuk metrics-metricsnya

Berdasarkan nilai recallnya (96%) apabila kita memiliki 100 orang nasabah yang resiko gagal bayar maka model Random Forest hanya dapat menangkap 96 orang diantaranya.

Akan tetapi model tersebut memiliki Overfit yang lebih dari model KNN sehingga kami memutuskan untuk menggunakan model Random Forest dengan dilakukan proses Hyperparameter Tuning untuk menangani Overfitting tersebut.

**Best Model Algoritma**

Kita memilih untuk menggunakan beberapa metric, antara lain yaitu Accuracy, Precision, Recall, & F1. Kemudian kita juga berfokus pada metric recall untuk mengurangi False Negative agar tidak terjadi hasil gagal bayar yang dimana model menganggap mampu membayar. Kemudian dari ketiga algoritma yang dibandingkan, tim kami memilih model Random Forest sebagai model terbaik.

![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/530fa258-487b-472f-bb0e-0714ae784e03)

**Threshold from Curve ROC**

Kami juga melakukan threshold pada rentang 0.1 hingga 0.9 pada kurva ROC. Hal ini dikarenakan kita perlu memilih kondisi terbaik dari metrics yang kita pilih khususnya recall. Berikut hasil threshold yang kami tampilkan dalam tabel.

![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/acdc7580-4a07-45c0-a5e3-e0f03988ed52)

Dari hasil uji Threshold tersebut, kami memilih model dengan threshold 0.4 karena selain nilai recall yang baik (0.93) model tersebut juga memiliki nilai Accurary, Precision, F1-Score yang baik sehingga secara keseluruhan threshold 0.4 memiliki nilai optimal.

**Feature Importance**

Setelah mendapatkan model yang paling baik dari RF + tuning, kami melakukan interpretasi dari hasil feature importance. Berikut grafik yang dihasilkan dari feature dengan skor nya berikut ini.

![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/9228167b-2e08-4313-a699-72bc0f532210)

- Business Insight: Lama bekerja di pekerjaan saat ini bisa menunjukkan stabilitas pekerjaan dan kepuasan dalam posisi tersebut dengan rata-rata lama bekerja selama 6 tahun dengan income rata-rata sebesar 5,000,449.
  Business Recommendation: Pertimbangkan lama bekerja di pekerjaan saat ini sebagai faktor dalam proses penilaian kredit. Nasabah yang telah lama bekerja di satu tempat mungkin diberi penawaran pinjaman yang lebih baik karena menunjukkan stabilitas yang lebih tinggi.

![image](https://github.com/wisnuph/pejuang-sabtu-malam/assets/64683758/58ad0f14-1426-4d77-bc51-8548b5e5cea9)

- Business Insight: Pendapatan nasabah merupakan salah satu faktor penting dalam memprediksi kemungkinan gagal bayar. Nasabah dengan pendapatan yang lebih tinggi cenderung memiliki pengalaman kerja yang lama, bekerja di pekerjaan saat ini dalam jangka waktu yang lama, sisa umur produktif kemungkinan tinggal sedikit, dan memiliki kemampuan lebih besar untuk membayar pinjaman mereka tepat waktu Rata-rata umur nasabah yang tidak tergolong dalam risk flag adalah 50 tahun.
  Business Recommendation: Dibutuhkan informasi nilai bunga untuk memitigasi resiko nasabah akan menunggak atau tidak dengan Perketat persyaratan pendapatan minimum untuk aplikasi pinjaman. Berikan pinjaman dengan bunga lebih rendah atau kondisi yang lebih baik kepada nasabah dengan pendapatan lebih tinggi untuk mengurangi risiko gagal bayar.

- Business Insight : Lamanya umur nasabah menunjukkan seberapa besar income yang didapatkan, lamanya bekerja di posisi saat ini, lamanya bekerja di perusahaan saat ini, serta jumlah usia produktif yang tersisa dari nasabah.
  Business Recommendation: Mempertimbangkan usia saat ini sebagai faktor dalam proses penilaian kredit. Nasabah yang usianya kisaran 15 - 64 tahun sebaiknya diberi penawaran pinjaman berdasarkan kelompok remaja (15- 25 tahun), dewasa (26 - 45), dan lansia (46 - 64) yang mengacu ke DepKes RI 2009 karena memudahkan proses penilaian kredit.

- Business Insight: Banyaknya sisa usia produktif dari nasabah menunjukkan seberapa besar income yang didapatkan, lamanya bekerja di posisi saat ini, lamanya bekerja di perusahaan saat ini, serta usia dari nasabah saat ini.
  Business Recommendation: Mempertimbangkan sisa usia produktif saat ini sebagai faktor dalam proses penilaian kredit. Nasabah yang sisa usia produktifnya kisaran 15 - 64 tahun sebaiknya diberi penawaran pinjaman karena menunjukkan masih memiliki usia yang produktif untuk bekerja.

## [Deployment](https://early-warning-loan-system.streamlit.app/)

Pada tahap terakhir ini kami mengembangkan suatu web dari model yang sudah dihasilkan agar dapat digunakan lebih user friendly untuk dicoba. Kami membuat dengan framework Streamlit dari library Python dan menyimpan & load model dengan Joblib. Kami juga menerapkan beberapa tampilan (UI) agar lebih menarik dan terdapat beberapa menu seperti Home, Tools, About Us, dan Contact. Di dalam web tersebut juga kami berikan cuplikan video tutorial agar dapat dilihat dan dipelajari terkait penggunaan dalam mengolah data (input, output, main tools, etc).
