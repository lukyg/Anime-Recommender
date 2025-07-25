# -*- coding: utf-8 -*-
"""Recommender System.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yIn6eujZVq_UYxZhNjAUDEs-OwoK7TF0

# **Recommender System: Anime Recommendation**

# **0. Domain Proyek**

Industri anime yang terus berkembang pesat dari waktu ke waktu memicu kebutuhan bagi pengguna untuk menemukan anime yang sesuai dengan preferensi mereka. Banyak pengguna yang merasa kesulitan dalam menentukan anime mana yang bagus untuk ditonton dari sekian banyak anime yang ada. Sistem rekomendasi menjadi solusi yang penting agar pengguna dapat dengan mudah menemukan anime favorit mereka berdasarkan rating atau preferensi mereka masing-masing.

Membangun sistem rekomendasi dapat meningkatkan pengalaman pengguna dalam menjelajah konten anime dan sekaligus membantu platform penyedia konten dalam meningkatkan engagement dan retensi pengguna. Dengan memanfaatkan data rating dan fitur anime, sistem ini bisa memberikan rekomendasi yang lebih personal.

**Referensi:**
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. Springer.

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.

# **1. Business Understanding**

## **1.1 Problem Statements**

Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem rekomendasi anime untuk menjawab permasalahan sebagai berikut:

- Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?

- Dengan data rating yang Anda miliki, bagaimana perusahaan dapat merekomendasikan anime lain yang mungkin disukai dan belum pernah dikunjungi oleh pengguna?

## **1.2 Goals**

Untuk menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Menghasilkan sejumlah rekomendasi anime yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
- Menghasilkan sejumlah rekomendasi anime yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya sebelumnya dengan teknik collaborative filtering.

## **1.3 Solution Approach**

Untuk mencapai tujuan tersebut, beberapa solusi yang akan dilakukan:

- Content-Based Filtering: Sistem rekomendasi yang menggunakan fitur anime (genre, type, episodes, rating, dsb) untuk merekomendasikan anime yang mirip dengan anime yang dipilih.

- Collaborative Filtering: Sistem rekomendasi yang menggunakan pola rating dari banyak pengguna untuk merekomendasikan anime yang disukai oleh pengguna lain dengan preferensi serupa.

# **2. Data Understanding**

## **2.1 Import Library**
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator
import statistics
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.optimizers import Adam

"""## **2.2 Dataset Preview**

- **Tentang Dataset**:

  Dataset ini berisi informasi tentang data preferensi pengguna dari 73.516 pengguna pada 12.294 anime. Setiap pengguna dapat menambahkan anime ke daftar putar mereka dan memberinya nilai berupa rating. Dataset ini merupakan gabungan dari rating-rating anime yang diberikan.

- **Key Feature**:

  1. Anime.csv

| **Nama Kolom** | **Deskripsi**                                                                 |
|----------------|---------------------------------------------------------------------------------|
| `anime_id`     | ID unik dari MyAnimeList untuk mengidentifikasi anime.                         |
| `name`         | Nama lengkap anime.                                                            |
| `genre`        | Daftar genre (dipisahkan koma) dari anime tersebut.                            |
| `type`         | Tipe anime (Movie, TV, OVA, dll).                                              |
| `episodes`     | Jumlah episode dari anime tersebut (1 jika berupa Movie).                      |
| `rating`       | Rata-rata rating dari anime (skala 1–10).                                      |
| `members`      | Jumlah anggota komunitas MyAnimeList yang memasukkan anime ini dalam grup mereka.|

  2. Rating.csv

| **Nama Kolom** | **Deskripsi**                                                                 |
|----------------|---------------------------------------------------------------------------------|
| `user_id`      | ID pengguna yang dihasilkan secara acak dan tidak dapat diidentifikasi.        |
| `anime_id`     | ID anime yang telah diberi rating oleh pengguna tersebut.                      |
| `rating`       | Rating dari pengguna (skala 1–10, atau `-1` jika ditonton tetapi tidak dirating).|

- **Sumber Dataset**:  
   Sumber dataset yang digunakan adalah **Anime Recommendation Database** dari repositori Kaggle dengan link berikut https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

## **2.3 Load Dataset**
"""

!curl -L -o /content/anime-recommendations-database.zip\
  https://www.kaggle.com/api/v1/datasets/download/CooperUnion/anime-recommendations-database

!unzip /content/anime-recommendations-database.zip

# Load anime.csv
animes = pd.read_csv('anime.csv')

animes.head()

animes.shape

animes.info()

# Load rating.csv
ratings = pd.read_csv('rating.csv')

ratings.head()

ratings.shape

ratings.info()

"""**Insight**:
- Dataset yang digunakan ada 2 yakni anime.csv dan rating.csv
- Pada dataset anime memiliki 7 kolom dengan 12294 baris data, 7 kolom terbagi menjadi 3 kolom numerik dan 4 kolom kategorik.
- Sedangkan pada dataset rating memiliki 3 kolom dengan 7813737 baris data yang seluruhnya bertipe data numerik.

## **2.4 Check Missing Value**
"""

animes.isna().any()

ratings.isna().any()

"""**Insight**:
- Pada dataset animes ditemukan nilai missing value pada kolom genre, type dan rating.

## **2.5 Check Duplicated Data**
"""

animes.duplicated().sum()

ratings.duplicated().sum()

"""**Insight**:
- Pada dataset anime tidak ditemukan nilai duplikat sedangkan pada dataset rating ditemukan 1 nilai duplikat. Sedangkan pada dataset rating tidak ditemukan nilai duplikat.

## **2.6 Check Unique Value**
"""

def unique_categorical(dataset, column_name):
    try:
        print(f"Number of unique {column_name}: {dataset[column_name].nunique()}")
        print(f"Unique {column_name}:")
        for value in dataset[column_name].unique():
            print(f"- {value}")
    except KeyError:
        print(f"Error: Column '{column_name}' not found in the dataset.")

unique_categorical(animes, 'name')

unique_categorical(animes, 'genre')

unique_categorical(animes, 'type')

unique_categorical(animes, 'episodes')

"""**Insight**:
- Pada kolom name memiliki nilai unik sebanyak 12292 dimana ini mewakili dari jumlah baris data dari dataset anime. Artinya setiap nilai unik mewakili setiap judul anime yang ada.
- Setiap anime memiliki kombinasi genre yang berbeda-beda, dengan jumlah kombinasi yang ada sebanyak 3264 data.
- Setiap anime juga terbagi menjadi 6 jenis seperti movie, OVA, dll.
- Terdapat 187 variasi jumlah episode yang unik, yang mengindikasikan bahwa banyak anime memiliki jumlah episode yang berbeda-beda, meskipun beberapa jumlah episode cukup sering muncul.

## **2.7 Exploratory Data Analysis (EDA)**

### **2.7.1 Descriptive Statistics**
"""

animes[['rating', 'members']].describe()

ratings['rating'].describe()

"""**Insight**:
- Sebagian besar anime dinilai cukup baik oleh pengguna, namun hanya sedikit yang mencapai rating sangat tinggi.
- Popularitas anime sangat bervariasi, dan mayoritas anime tidak memiliki banyak penonton.
- Pada dataset rating memiliki nilai -1, artinya anime tersebut sudah ditonton pengguna namun belum diberikan rating penilaian.

### **2.7.2 Univariat Analysis**
"""

# number of ratings
print("There are", str(len(ratings)), "rating posted.")

# number of users
print("From total", str(len(ratings['user_id'].unique())), "unique user id.")

# number of unique animes (in anime list, not ratings)
print("And rate total", str(len(animes['anime_id'].unique())), "different animes.")
# avg number of anime rated per user

# distribution of ratings posted per user
ratings_per_user = ratings.groupby('user_id')['rating'].count()
statistics.mean(ratings_per_user.tolist())

# distribution of ratings per user
_ = plt.title('Distribution of Ratings Posted per User')
_ = sns.histplot(ratings_per_user)
plt.show()

"""**Insight**:
- Dari grafik distribusi menunjukkan bahwa rating per pengguna yang tidak merata dan cenderung right-skewed, dengan mayoritas pengguna memberikan sedikit rating. Hal ini menandakan rendahnya partisipasi pengguna dalam memberikan rating dan hal ini dapat menjadi tantangan dalam membuat sistem rekomendasi.
"""

# avg number of ratings given per anime
ratings_per_anime = ratings.groupby('anime_id')['rating'].count()
statistics.mean(ratings_per_anime.tolist())

# distribution of ratings per anime
_ = plt.title('Distribution of Ratings Posted per Anime')
_ = sns.histplot(ratings_per_anime)
plt.show()

"""**Insight**:
- Grafik rating per anime menunjukkan distribusi right-skewed, di mana sebagian besar anime hanya menerima sedikit rating. Ini mengindikasikan bahwa hanya sebagian kecil anime yang populer dan sering dinilai, sementara sebagian besar lainnya jarang mendapat perhatian dari pengguna.

# **3. Data Preparation**

## **3.1 Clean Missing Value**
"""

animes.isna().sum()

animes.dropna(inplace=True)

animes.isna().sum()

"""**Insight**:
- Dalam tahap data preparation, dilakukan pembersihan data dengan menghapus data yang memiliki missing values. Ditemukan bahwa kolom genre memiliki 62 data yang hilang, type sebanyak 25, dan rating sebanyak 230. Karena ketiga kolom ini merupakan informasi penting dalam analisis dan sistem rekomendasi, seluruh baris yang mengandung nilai kosong pada kolom tersebut dihapus dari dataset.
- Proses ini dilakukan dengan tujuan untuk menjaga konsistensi dan kualitas data, serta menghindari gangguan dalam proses eksplorasi, visualisasi, maupun pemodelan. Dengan data yang bersih, analisis menjadi lebih akurat dan model tidak salah dalam memberikan rekomendasi.

## **3.2 Clean Duplicated**
"""

duplicate_rows = ratings[ratings.duplicated()]
ratings.drop_duplicates(inplace=True)
print(duplicate_rows)

"""**Insight**:
- Hanya 1 duplicated data yand ditemukan pada dataset ratings dan sudah dibersihkan.
- Menghapus data duplikat penting dengan tujuan untuk memastikan bahwa setiap interaksi pengguna dengan anime bersifat unik. Duplikasi dapat menyebabkan bias dalam analisis, seperti menghitung rating lebih dari satu kali, yang pada akhirnya dapat memengaruhi hasil visualisasi, evaluasi, atau performa model sistem rekomendasi.

## **3.3 Remove Rating with -1 Value**
"""

# remove rating -1
ratings = ratings[ratings["rating"] != -1]

ratings.groupby(["rating"])["rating"].count()

"""**Insight**:
- Rating bernilai -1 merepresentasikan bahwa pengguna belum memberikan penilaian terhadap anime yang telah mereka tonton. Karena rating yang valid berada pada rentang 1 hingga 10, maka nilai -1 dianggap tidak relevan dan perlu dihapus dari dataset.
- Dilakukan identifikasi terhadap entri rating bernilai -1 pada dataset ratings. Seluruh entri dengan nilai ini difilter dan dihapus menggunakan operasi seleksi pada DataFrame, agar hanya menyisakan rating yang benar-benar diberikan oleh pengguna.
- Menghapus nilai -1 penting agar hanya rating yang valid dan bermakna yang digunakan dalam proses analisis dan pembuatan model rekomendasi. Jika nilai -1 tetap disertakan, model dapat salah mempelajari preferensi pengguna dan menghasilkan rekomendasi yang kurang akurat. Pembersihan ini juga meningkatkan kualitas dan keandalan dataset secara keseluruhan.

## **3.4 TF-IDF Vectorizer**
"""

tfidf = TfidfVectorizer(stop_words='english')
animes['genre'] = animes['genre'].fillna('')
tfidf_matrix = tfidf.fit_transform(animes['genre'])

tfidf_matrix.shape

tfidf_matrix.todense()

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names_out(),
    index=animes['name']
).sample(22, axis=1).sample(10, axis=0)

"""**Insight**:
- Dari hasil analisis TF-IDF pada genre anime, ditemukan bahwa genre-genre yang jarang muncul seperti hentai memiliki nilai TF-IDF yang tinggi, menunjukkan bahwa genre tersebut bersifat khas dan mampu membedakan anime tertentu, seperti Soukou Kijo Iris, dari anime lainnya. Sebaliknya, genre populer seperti school atau romance cenderung memiliki nilai TF-IDF rendah karena terlalu umum dan sering muncul di berbagai judul, sehingga kurang memberikan identitas yang unik. Beberapa anime seperti seri Dirty Pair memperlihatkan konsistensi dalam genre utamanya, seperti police, yang memperkuat identitas tematik dan mempermudah sistem dalam mengenali pola genre. Di sisi lain, anime dengan kombinasi genre yang unik seperti Jigoku Sensei Nube memiliki profil genre yang lebih khas, menjadikannya lebih mudah untuk direkomendasikan secara akurat oleh sistem.

- Proses representasi ini penting agar informasi genre dapat diubah ke dalam bentuk numerik yang dapat dipahami oleh algoritma, khususnya dalam sistem rekomendasi berbasis konten (content-based filtering). Dengan menggunakan pendekatan TF-IDF, genre yang jarang namun bermakna dapat lebih ditonjolkan, sehingga membantu sistem dalam membedakan satu anime dengan yang lain. Hal ini berkontribusi pada peningkatan relevansi dan akurasi hasil rekomendasi. Tanpa proses ini, algoritma cenderung memperlakukan semua genre secara setara, sehingga kesulitan dalam mengenali ciri khas yang melekat pada masing-masing anime.

## **3.5 Cossine Similarity**
"""

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim

# Buat DataFrame cosine similarity matrix
cosine_sim_df = pd.DataFrame(cosine_sim, index=animes['name'], columns=animes['name'])

print('Shape:', cosine_sim_df.shape)

# Cek sebagian isi matrix
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""**Insight**:
- Berdasarkan hasil perhitungan cosine similarity antar anime, ditemukan bahwa beberapa pasangan anime memiliki tingkat kemiripan genre yang sangat tinggi. Contohnya, Picchipichi Shizuku-chan dan Momon's Sand Witch Episode 0 menunjukkan nilai kemiripan sempurna (1.0), yang menandakan bahwa keduanya memiliki profil genre yang hampir identik dan sangat cocok untuk direkomendasikan satu sama lain. Di sisi lain, anime seperti Hidamari no Ki dan Nido to Mezamenu Komori Uta memiliki nilai cosine similarity mendekati nol, yang mencerminkan bahwa genre keduanya sangat berbeda dan tidak memiliki kesamaan yang cukup untuk dijadikan referensi dalam sistem rekomendasi berbasis genre.

- Tahapan ini penting dalam membangun sistem rekomendasi berbasis konten (content-based filtering), di mana rekomendasi diberikan berdasarkan kemiripan fitur dalam hal ini, genre. Penggunaan cosine similarity memungkinkan pengukuran tingkat kesamaan antar anime secara efisien, meskipun vektor genre berada dalam ruang berdimensi tinggi. Melalui pendekatan ini, sistem mampu mengidentifikasi dan merekomendasikan anime dengan tema atau nuansa serupa, sehingga menghasilkan rekomendasi yang lebih relevan dan sesuai dengan preferensi pengguna.

## **3.6 Feature Engineering**
"""

ratings_per_anime = ratings.groupby('anime_id').count()['rating']
ratings_per_user = ratings.groupby('user_id').count()['rating']

popular_anime_ids = ratings_per_anime[ratings_per_anime >= 1000].index
popular_user_ids = ratings_per_user[ratings_per_user >= 500].index

filtered_ratings = ratings[
    ratings['anime_id'].isin(popular_anime_ids) &
    ratings['user_id'].isin(popular_user_ids)
].copy()

filtered_ratings

"""**Insight**:
- Dari hasil feature engineering, hanya anime yang telah menerima ≥ 1000 rating dan user yang telah memberikan ≥ 500 rating yang dipertahankan dalam dataset. Hal ini menghasilkan subset data dengan interaksi yang lebih padat (dense), menunjukkan bahwa hanya anime populer dan user aktif yang dilibatkan dalam proses pelatihan model. Hal ini penting untuk memastikan kualitas rekomendasi tetap tinggi dan mengurangi kemungkinan noise dari data yang jarang berinteraksi.
- Tahapan ini diperlukan untuk mengurangi sparsity pada data, di mana sebagian besar user tidak memberikan rating ke semua anime, sehingga dengan memfilter hanya user aktif dan anime populer, data menjadi lebih padat dan representatif. Hal ini membantu model lebih mudah mengenali pola interaksi, menghindari noise dari data yang minim informasi, serta meningkatkan akurasi dan efisiensi model. Selain itu, penyederhanaan ini juga mempercepat proses pelatihan tanpa mengorbankan kualitas rekomendasi.

## **3.7 Label Encoding**
"""

user_enc = LabelEncoder()
anime_enc = LabelEncoder()

filtered_ratings['user'] = user_enc.fit_transform(filtered_ratings['user_id'].values)
filtered_ratings['anime'] = anime_enc.fit_transform(filtered_ratings['anime_id'].values)

num_users = filtered_ratings['user'].nunique()
num_animes = filtered_ratings['anime'].nunique()

"""**Insight**:
- Proses label encoding berhasil mengubah user_id dan anime_id menjadi format numerik, menghasilkan total {num_users} pengguna unik dan {num_animes} anime unik dalam data yang telah difilter. Ini memungkinkan data digunakan secara efisien dalam model neural network yang hanya menerima input numerik.
- Tahapan label encoding diperlukan karena model machine learning, khususnya neural network seperti recommenderNet, hanya dapat memproses input dalam bentuk numerik. Identifier seperti user_id atau anime_id yang berupa string atau angka tidak berurutan tidak dapat digunakan langsung sebagai input embedding. Oleh karena itu, encoding ini mengubah ID menjadi indeks numerik berurutan, sehingga dapat dipetakan ke dalam embedding layer dan diproses oleh model. Proses ini juga memastikan efisiensi dan konsistensi dalam representasi data saat pelatihan model rekomendasi.

## **3.8 Split Train & Test**
"""

x = filtered_ratings[['user', 'anime']].values
y = filtered_ratings['rating'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

"""**Insight**:
- Dataset dibagi menjadi dua bagian, yaitu 80% data untuk pelatihan (train set) dan 20% untuk pengujian (test set). Pembagian ini bertujuan untuk mengevaluasi performa model terhadap data yang belum pernah dilihat sebelumnya, guna menghindari overfitting dan memastikan generalisasi model.

- Split data menjadi train dan test sangat penting untuk mengukur kemampuan generalisasi model. Tanpa data uji yang terpisah, model bisa hanya “menghafal” data pelatihan (overfitting) dan gagal memprediksi dengan baik pada data baru. Dengan membagi dataset, kita dapat mengevaluasi seberapa baik model belajar dari data pelatihan dan seberapa akurat prediksinya terhadap data yang tidak dikenal. Penggunaan parameter random_state=42 juga memastikan bahwa pembagian data bersifat reproducible.

# **4. Modeling and Result**

Dalam proyek ini, sistem rekomendasi anime dikembangkan menggunakan dua pendekatan utama: **Content-based Filtering** dan **Collaborative Filtering**. Berikut adalah penjelasan rinci untuk masing-masing pendekatan.

## **Content-based Filtering**

Sistem ini merekomendasikan anime berdasarkan kemiripan konten, khususnya **genre**, dengan anime yang pernah disukai oleh pengguna. Dengan proses kerja sebagai berikut.
1. Genre anime diolah menggunakan **TF-IDF Vectorizer** untuk mengubah data teks menjadi vektor numerik.
2. **Cosine Similarity** digunakan untuk menghitung kemiripan antar anime berdasarkan vektor genre.
3. Anime dengan nilai kemiripan tertinggi terhadap anime input (misalnya, *Naruto*) akan direkomendasikan.

Dengan input *Naruto*, sistem merekomendasikan anime yang relevan seperti:
- *Naruto: Shippuuden*
- *Boruto*
- *Naruto Movie Series*

Sistem ini mampu mengenali kesamaan genre seperti *Action*, *Comedy*, *Shounen*, *Martial Arts*, dan *Super Power*.

### **Kelebihan:**
- Dapat merekomendasikan item baru tanpa data interaksi pengguna.
- Lebih mudah dijelaskan secara interpretatif (berdasarkan genre).

### **Kekurangan:**
- Kurang personalisasi.
- Rekomendasi cenderung terlalu mirip atau repetitif.

## **Collaborative Filtering**

Sistem ini merekomendasikan anime berdasarkan **pola interaksi** antara pengguna dan anime, tanpa mempertimbangkan konten anime itu sendiri. Model yang digunakan adalah **RecommenderNet**. Dengan cara kerja sebagai berikut.


1. Setiap pengguna dan anime direpresentasikan sebagai **vektor embedding**.
2. Hasil rekomendasi dihasilkan berdasarkan **dot product** antara vektor pengguna dan vektor anime.
3. Model memberikan rekomendasi anime yang sesuai dengan preferensi pengguna tertentu.

### **Kelebihan:**
- Memberikan rekomendasi personal berdasarkan preferensi pengguna lain.
- Mampu menangkap pola tersembunyi di luar genre.

### **Kekurangan:**
- Membutuhkan data interaksi yang banyak (masalah *sparse matrix*).
- Tidak dapat merekomendasikan anime baru (*cold start item*).

## **4.1 Content-Based Filtering**
"""

indices = pd.Series(animes.index, index=animes['name']).drop_duplicates()

def content_based_recommendation(title, top_n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    anime_indices = [i[0] for i in sim_scores]
    return animes[['name', 'genre']].iloc[anime_indices]

content_based_recommendation("Naruto")

"""**Insight**:
- Berdasarkan hasil content-based recommendation dengan input anime berjudul Naruto, sistem merekomendasikan 10 anime lain yang mayoritas berasal dari seri yang sama, seperti Naruto: Shippuuden, Boruto, dan berbagai film spin-off. Hal ini menunjukkan bahwa sistem berhasil mengidentifikasi kesamaan konten berdasarkan genre, di mana semua anime yang direkomendasikan memiliki genre yang sangat mirip, yaitu Action, Comedy, Martial Arts, Shounen, dan Super Power.

## **4.2 Collaborative Filtering**
"""

embedding_size = 25

# Input
user_input = Input(shape=(1,))
anime_input = Input(shape=(1,))

# Embedding
user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
anime_embedding = Embedding(num_animes, embedding_size, name='anime_embedding')(anime_input)

# Dot product
dot_product = Dot(axes=2)([user_embedding, anime_embedding])
dot_product = Flatten()(dot_product)

output = Dense(1)(dot_product)

model = Model(inputs=[user_input, anime_input], outputs=output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model.summary()

history = model.fit(
    [x_train[:, 0], x_train[:, 1]],
    y_train,
    validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
    batch_size=64,
    epochs=20,
    verbose=1
)

def recommend_for_user(user_id, animes, ratings, model, top_n=10):
    if user_id not in user_enc.classes_:
        raise ValueError(f"user_id {user_id} tidak dikenal oleh encoder.")

    # Ambil semua anime_id
    all_anime_ids = animes['anime_id'].unique()

    # Ambil anime yang sudah ditonton user
    watched_anime_ids = ratings[ratings['user_id'] == user_id]['anime_id'].values

    # Filter hanya anime yang belum ditonton user
    not_watched_anime_ids = np.setdiff1d(all_anime_ids, watched_anime_ids)

    # Filter anime_id yang dikenal encoder
    not_watched_anime_ids = [aid for aid in not_watched_anime_ids if aid in anime_enc.classes_]

    # Encode user dan anime
    user_encoded = user_enc.transform([user_id] * len(not_watched_anime_ids))
    anime_encoded = anime_enc.transform(not_watched_anime_ids)

    # Prediksi rating
    predictions = model.predict([user_encoded, anime_encoded], verbose=0)

    # Gabungkan dengan anime_id
    pred_df = pd.DataFrame({
        'anime_id': not_watched_anime_ids,
        'predicted_rating': predictions.flatten()
    })

    # Gabung dengan data anime
    top_recs = pred_df.merge(animes, on='anime_id')
    top_recs = top_recs.sort_values('predicted_rating', ascending=False).head(top_n)

    return top_recs[['anime_id', 'name', 'genre', 'predicted_rating']]

selected_user_id = 226
print("User ID yang dipilih:", selected_user_id)

recommendations = recommend_for_user(
    user_id=selected_user_id,
    animes=animes,
    ratings=filtered_ratings,
    model=model
)

recommendations

"""**Insight**:
- Menggunakan Neural Collaborative Filtering (recommenderNet) dengan embedding berukuran 25 untuk user dan anime, serta dot product yang dilanjutkan dengan layer Dense sebagai output.
- Model dilatih menggunakan Mean Squared Error (MSE) sebagai loss function dan optimizer Adam dengan learning rate 0.001. Proses pelatihan dilakukan selama 20 epoch dengan batch size 64, serta validasi menggunakan data test.
- Hasil rekomendasi dari model Collaborative Filtering untuk user ID 226 menunjukkan bahwa sistem mampu mengenali preferensi pengguna dengan baik, terbukti dari variasi genre seperti action, romance, sci-fi, dan comedy yang muncul dalam daftar top-10. Model merekomendasikan anime populer dan berkualitas tinggi seperti Kimi no Na wa., Redline, dan Fate/stay night: UBW, dengan prediksi rating tinggi dan konsisten di kisaran 8.6–9.1. Hal ini menunjukkan tingkat keyakinan model yang baik serta kemampuan personalisasi yang kuat. Selain itu, rekomendasi terhadap anime dengan gaya dan tema unik seperti Interstella5555 dan Afro Samurai mengindikasikan bahwa model mampu menangkap selera pengguna secara lebih mendalam, tidak hanya berdasarkan popularitas umum, tetapi juga dari pola interaksi tersembunyi antar pengguna.

# **5. Evaluation**

Dalam proyek ini, menggunakan metrik Precision@K untuk mengukur performa dari content-based filtering. Sedangkan untuk colaborative filtering menggunakan metrik MSE (Mean Squared Error) yang diuraikan sebagai berikut.

### Precision@K

- **Formula**:  
  $$
  Precision@K = \frac{|\text{Recommended}_K \cap \text{Relevant}|}{K}
  $$

- **Penjelasan**:  
  Precision@K mengukur **proporsi item relevan** di antara *K* item teratas yang direkomendasikan oleh sistem.  
  Nilai Precision@K yang **lebih tinggi** menunjukkan bahwa sistem berhasil merekomendasikan lebih banyak anime yang relevan kepada pengguna.  
  Metrik ini penting karena dalam praktiknya, pengguna biasanya hanya memperhatikan sejumlah kecil rekomendasi teratas (misalnya top-5 atau top-10).  
  Precision@K membantu mengukur **kualitas rekomendasi** pada posisi yang paling penting bagi pengguna.

### MSE (Mean Squared Error)

- **Formula**:
  $$
  MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
  $$

- **Penjelasan**:  
  MSE mengukur **rata-rata kuadrat selisih** antara nilai aktual ($y_i$) dan nilai prediksi ($\hat{y}_i$).  
  Nilai MSE yang **lebih kecil** menandakan bahwa prediksi model semakin mendekati nilai aktual.  
  Karena menggunakan **kuadrat dari selisih**, MSE akan memberikan penalti lebih besar untuk kesalahan prediksi yang besar, sehingga sensitif terhadap outlier.
"""

ground_truth = animes[animes['rating'] > 7][['name', 'genre']]

ground_truth

def evaluate_recommendation(title, top_n=10):
    # Dapatkan rekomendasi
    recommendations = content_based_recommendation(title, top_n)
    recommended_titles = set(recommendations['name'])

    # Dapatkan ground truth (misalnya, anime dengan rating > 7 untuk pengguna)
    relevant_animes = set(ground_truth['name'])  # Sesuaikan dengan data ground truth

    # Hitung jumlah rekomendasi yang relevan
    relevant_recommendations = recommended_titles.intersection(relevant_animes)

    # Precision: relevan direkomendasikan / total rekomendasi
    precision = len(relevant_recommendations) / len(recommended_titles) if recommended_titles else 0

    return precision

# Contoh penggunaan
title = "Naruto"  # Ganti dengan judul anime
precision = evaluate_recommendation(title, top_n=10)
print(f"Precision: {precision:.2f}")

"""**Insight**:
- Dengan menggunakan anime Naruto sebagai acuan, model menghasilkan nilai Precision@10 sebesar 0.90, yang berarti 9 dari 10 anime yang direkomendasikan kepada pengguna benar-benar relevan dengan preferensi genre atau tema yang serupa. Nilai ini mencerminkan bahwa sistem rekomendasi berbasis konten mampu menyajikan daftar rekomendasi yang sangat tepat sasaran, meminimalkan item tidak relevan, dan meningkatkan kemungkinan pengguna tertarik pada rekomendasi yang diberikan. Tingginya precision ini menunjukkan efektivitas model dalam memahami karakteristik unik Naruto dan mencocokkannya dengan anime lain yang sejenis, sehingga meningkatkan kualitas pengalaman pengguna secara keseluruhan.
"""

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

train_mse = history.history['loss'][-1]
val_mse = history.history['val_loss'][-1]

print("------- MSE (Mean Squared Error) --------")
print(f"Train MSE: {train_mse:.4f}")
print(f"Validation MSE: {val_mse:.4f}")

"""**Insight**:
- Berdasarkan grafik latih dan nilai MSE, model RecommenderNet menunjukkan penurunan signifikan pada training loss dari 21.8 menjadi 0.85 dan validation loss dari 1.43 menjadi 1.18 dalam 7–8 epoch pertama, yang menandakan proses pembelajaran awal berjalan efektif dan model berhasil mengurangi error secara drastis.

- Stabilnya validation MSE di kisaran 1.17–1.18 pada epoch akhir juga menunjukkan bahwa kapasitas model sudah optimal.

- Selisih yang kecil antara Train MSE (0.8548) dan Validation MSE (1.1923) mengindikasikan bahwa model memiliki kemampuan generalisasi yang baik terhadap data baru, serta tidak menunjukkan gejala overfitting.

## **5.1 Dampak terhadap Business**

Berdasarkan hasil yang dibuat, berikut hasil berdasarkan latar belakang masalah.
- Penerapan content-based filtering dan collaborative filtering dengan model RecommenderNet memberikan dampak signifikan terhadap pemahaman dan solusi bisnis dalam industri anime. Content-based filtering memungkinkan sistem untuk merekomendasikan anime yang mirip berdasarkan fitur seperti genre, skor, dan episode, sehingga tetap relevan dengan preferensi pengguna yang sudah diketahui. Di sisi lain, collaborative filtering dengan RecommenderNet memberikan pendekatan personalisasi lebih dalam dengan mempelajari pola rating antar pengguna melalui embedding layer, yang menghasilkan top-10 rekomendasi dengan prediksi rating tinggi, misalnya Kimi no Na wa., Redline, hingga Afro Samurai. Evaluasi dengan metrik MSE menunjukkan nilai kesalahan prediksi yang cukup rendah, dengan grafik latih dan validasi yang stabil hingga epoch ke-20, menandakan model telah belajar secara efektif tanpa overfitting. Dampaknya, sistem rekomendasi ini meningkatkan pengalaman pengguna secara signifikan dalam menemukan anime yang sesuai preferensi mereka, sekaligus memberikan nilai tambah bagi platform penyedia konten dalam meningkatkan engagement, durasi menonton, dan loyalitas pengguna secara keseluruhan.

## **5.2 Refleksi terhadap Problem Statement dan Goals**

Laporan ini telah menjawab kedua problem statement dan memenuhi goals yang ditetapkan pada tahap business understanding:

- Problem Statement 1: “Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?”
Masalah ini telah dijawab dengan membangun sistem content-based filtering yang merekomendasikan anime berdasarkan kemiripan fitur (nama dan genre) terhadap anime yang disukai pengguna.

- Problem Statement 2: “Dengan data rating yang Anda miliki, bagaimana perusahaan dapat merekomendasikan anime lain yang mungkin disukai dan belum pernah dikunjungi oleh pengguna?”
Permasalahan ini ditangani melalui collaborative filtering berbasis embedding neural network, yang mampu menghasilkan rekomendasi yang dipersonalisasi berdasarkan pola rating pengguna lain yang memiliki preferensi serupa.

- **Goal**: Membangun sistem rekomendasi anime yang dipersonalisasi menggunakan content-based dan collaborative filtering.  
Tujuan ini telah tercapai dengan membangun dua model rekomendasi yaitu pertama menggunakan pendekatan **content-based filtering** berbasis fitur anime, dan kedua menggunakan **collaborative filtering** berbasis interaksi pengguna (rating). Model collaborative filtering yang dibangun dengan arsitektur **RecommenderNet** menunjukkan performa yang cukup baik dengan nilai **MSE sebesar 0.8548 untuk data latih** dan **1.1923 untuk data validasi**, mengindikasikan kemampuan generalisasi yang baik tanpa overfitting. Model ini berhasil menghasilkan daftar **top-10 anime dengan skor prediksi tinggi** untuk pengguna tertentu (contoh: user **226**), di antaranya **Kimi no Na wa.**, **Redline**, dan **Afro Samurai**, yang menunjukkan bahwa sistem mampu memberikan rekomendasi yang populer, relevan, dan sesuai dengan preferensi pengguna.

Oleh karena itu, proyek ini tidak hanya menyelesaikan tantangan teknis dalam membangun sistem rekomendasi berbasis content-based dan collaborative filtering, namun juga berhasil memberikan solusi yang berdampak nyata terhadap proses bisnis platform penyedia konten anime. Sistem rekomendasi yang dibangun mampu meningkatkan personalisasi, mendorong interaksi pengguna, serta membantu pengguna dalam menemukan konten yang relevan dengan preferensi mereka secara efisien. Hal ini mendukung peningkatan kepuasan pengguna dan retensi platform, sejalan dengan tujuan utama dari proyek ini.
"""