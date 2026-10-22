"""Projenin her yerinde kullanılan sabitler: dosya yolları, sütun isimleri, hedef, rastgelelik tohumu."""  # bu modülün amacını anlatan açıklama

from pathlib import Path  # dosya yollarını Windows/Mac/Linux fark etmeksizin aynı şekilde yönetmemizi sağlayan Path sınıfını içe aktarır

# --- Dosya yolları ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # bu dosyanın (src/mlfi/config.py) iki üst klasörüne çıkarak proje kök klasörünü bulur
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"  # ham veri dosyalarının durduğu klasörün yolu (data/raw)
TRAIN_PATH = RAW_DATA_DIR / "adult.data"  # eğitim dosyasının tam yolu
TEST_PATH = RAW_DATA_DIR / "adult.test"  # test dosyasının tam yolu

# --- Sütunlar (adult.names dosyasındaki sırayla; Python'da '-' kullanılamadığı için '_' yaptık) ---
NUMERIC_COLS = [  # sayısal ("continuous") sütunların listesi; drift'te bunlara KS testi uygulanacak
    "age",  # yaş
    "fnlwgt",  # anket örneklem ağırlığı (kişinin özelliği değil! modele girip girmeyeceğini ileride tartışacağız)
    "education_num",  # eğitim seviyesinin sayısal kodu
    "capital_gain",  # sermaye kazancı
    "capital_loss",  # sermaye kaybı
    "hours_per_week",  # haftalık çalışma saati
]  # sayısal sütun listesinin sonu

CATEGORICAL_COLS = [  # kategorik (seçenekli) sütunların listesi; drift'te bunlara ki-kare testi uygulanacak
    "workclass",  # çalışma sınıfı (Private, State-gov...)
    "education",  # eğitim seviyesi (yazıyla)
    "marital_status",  # medeni durum
    "occupation",  # meslek
    "relationship",  # hanedeki rol
    "race",  # ırk (alt grup analizinde kullanılacak)
    "sex",  # cinsiyet (alt grup analizinde kullanılacak)
    "native_country",  # doğduğu ülke
]  # kategorik sütun listesinin sonu

TARGET = "income"  # tahmin etmeye çalıştığımız hedef sütunun adı (dosyada ismi yoktu, biz verdik)
POSITIVE_LABEL = ">50K"  # "pozitif sınıf": yılda 50 binden fazla kazananlar; precision/recall bu sınıfa göre hesaplanacak
VALID_LABELS = {"<=50K", ">50K"}  # temizlenmiş veride hedef sütunun alabileceği tek iki değer

COLUMN_NAMES = [  # ham dosyadaki sütunların SOLDAN SAĞA gerçek sırası (read_csv bu sırayla isim verecek)
    "age",  # 1. sütun
    "workclass",  # 2. sütun
    "fnlwgt",  # 3. sütun
    "education",  # 4. sütun
    "education_num",  # 5. sütun
    "marital_status",  # 6. sütun
    "occupation",  # 7. sütun
    "relationship",  # 8. sütun
    "race",  # 9. sütun
    "sex",  # 10. sütun
    "capital_gain",  # 11. sütun
    "capital_loss",  # 12. sütun
    "hours_per_week",  # 13. sütun
    "native_country",  # 14. sütun
    TARGET,  # 15. sütun: hedef (income)
]  # sütun isimleri listesinin sonu

RANDOM_SEED = 42  # bütün rastgele işlemlerde (bölme, hata enjeksiyonu) kullanılacak sabit tohum; deneyler tekrarlanabilir olsun diye
