import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ==================================================
# 1. LOAD DATA
# ==================================================
try:
    df = pd.read_csv("spotify_data clean.csv")
    print("✅ Data berhasil dimuat.")
except FileNotFoundError:
    print("❌ File tidak ditemukan.")
    exit()

# ==================================================
# 2. PREPROCESSING
# ==================================================
df = df.drop_duplicates()
df = df[df['artist_genres'] != 'N/A'].copy()

df['artist_popularity'] = df['artist_popularity'].fillna(df['artist_popularity'].mean())
df['artist_followers'] = df['artist_followers'].fillna(df['artist_followers'].mean())

# ==================================================
# 3. EDA – GRAFIK TAMPIL
# ==================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['track_popularity'], bins=30, kde=True)
plt.title('Distribusi Popularitas Lagu')

plt.subplot(1, 2, 2)
top_genres = df['artist_genres'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title('10 Genre Terbanyak')

plt.tight_layout()
plt.show(block=True)
plt.close()

# ==================================================
# 4. TRANSFORMASI & ENCODING
# ==================================================
df['target'] = df['track_popularity'].apply(lambda x: 1 if x > 50 else 0)

le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['artist_genres'].astype(str))

# ==================================================
# 5. SPLIT DATA & SCALING
# ==================================================
features = [
    'artist_popularity',
    'artist_followers',
    'track_duration_min',
    'genre_encoded'
]

X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# 6. PEMODELAN
# ==================================================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# ==================================================
# 7. EVALUASI (TEKS)
# ==================================================
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nAkurasi Decision Tree : {acc_dt:.2%}")
print(f"Akurasi Random Forest : {acc_rf:.2%}")

print("\nLaporan Klasifikasi (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# ==================================================
# 8. TABEL HASIL EVALUASI (TAMPIL)
# ==================================================
eval_df = pd.DataFrame({
    'Algoritma': ['Decision Tree', 'Random Forest'],
    'Accuracy': [acc_dt, acc_rf]
})

plt.figure(figsize=(6, 2))
plt.axis('off')

plt.table(
    cellText=np.round(eval_df[['Accuracy']].values, 4),
    colLabels=['Accuracy'],
    rowLabels=eval_df['Algoritma'],
    loc='center'
)

plt.title('Tabel Hasil Evaluasi Model')
plt.show(block=True)
plt.close()

# ==================================================
# 9. CONFUSION MATRIX (TAMPIL)
# ==================================================
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')

plt.show(block=True)
plt.close()
