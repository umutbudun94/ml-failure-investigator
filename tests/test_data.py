"""clean() fonksiyonunun testleri. Gerçek veriye ihtiyaç duymaz; Adult formatında küçük sahte dosyalar kullanır."""  # test dosyasının amacını anlatır

import pandas as pd  # DataFrame işlemleri ve tip kontrolleri için pandas'ı içe aktarır
import pytest  # test altyapısını (fixture, raises) içe aktarır

from mlfi.config import CATEGORICAL_COLS, COLUMN_NAMES, NUMERIC_COLS, TARGET, VALID_LABELS  # testlerde kullanılacak sabitleri alır
from mlfi.data import clean, load_raw  # test edilecek fonksiyonları alır

TRAIN_TEXT = (  # adult.data dosyasının biçimini taklit eden küçük örnek metin
    "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n"  # normal satır
    "54, ?, 180211, Some-college, 10, Married-civ-spouse, ?, Husband, Asian-Pac-Islander, Male, 0, 0, 60, South, >50K\n"  # eksik değer içeren satır
    "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n"  # ilk satırın tıpatıp aynısı (yinelenen kayıt)
)  # eğitim örneğinin sonu

TEST_TEXT = (  # adult.test dosyasının biçimini taklit eden küçük örnek metin
    "|1x3 Cross validator\n"  # veri olmayan açıklama satırı
    "25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, Own-child, Black, Male, 0, 0, 40, United-States, <=50K.\n"  # test dosyası tarzı etiketli satır
    "38, Private, 89814, HS-grad, 9, Married-civ-spouse, Farming-fishing, Husband, White, Male, 0, 0, 50, ?, >50K.\n"  # hem eksik değer hem test tarzı etiket
)  # test örneğinin sonu


@pytest.fixture  # bu fonksiyonun testlere hazır veri sağlayan bir "fixture" olduğunu belirtir
def raw_train(tmp_path):  # pytest'in verdiği geçici klasörde sahte eğitim dosyası oluşturan fixture
    path = tmp_path / "adult.data"  # geçici dosyanın yolu
    path.write_text(TRAIN_TEXT)  # örnek metni dosyaya yazar
    return load_raw(path)  # dosyayı ham olarak okuyup testlere verir


@pytest.fixture  # bir fixture daha tanımlar
def raw_test(tmp_path):  # sahte test dosyası oluşturan fixture
    path = tmp_path / "adult.test"  # geçici dosyanın yolu
    path.write_text(TEST_TEXT)  # örnek metni dosyaya yazar
    return load_raw(path)  # dosyayı ham olarak okuyup testlere verir


# ---------- load_raw (hazır) ----------

def test_load_raw_column_names(raw_train):  # okunan verinin sütun isimlerini kontrol eden test
    assert list(raw_train.columns) == COLUMN_NAMES  # sütunlar config'teki sırayla ve isimlerle gelmeli


def test_load_raw_missing_file(tmp_path):  # olmayan dosya verilince ne olduğunu kontrol eden test
    with pytest.raises(FileNotFoundError):  # bu bloğun FileNotFoundError fırlatmasını bekler
        load_raw(tmp_path / "yok.data")  # var olmayan bir dosyayı okumaya çalışır


# ---------- clean (GÖREV 1) ----------

def test_clean_strips_whitespace(raw_train):  # şart 1: boşluklar temizlenmiş mi
    df = clean(raw_train)  # ham veriyi temizler
    for col in CATEGORICAL_COLS + [TARGET]:  # her metin sütununu tek tek gezer
        values = df[col].dropna()  # eksik olmayan değerleri alır
        assert (values == values.str.strip()).all(), f"{col} sütununda boşluk kalmış"  # her değer kenar boşluksuz olmalı


def test_clean_marks_missing_as_nan(raw_train):  # şart 2: eksik işareti NaN'a çevrilmiş mi
    df = clean(raw_train)  # ham veriyi temizler
    assert not (df == "?").any().any(), "Veride hâlâ '?' var"  # tabloda hiçbir yerde '?' kalmamalı
    assert df["workclass"].isna().sum() == 1  # örnekte workclass'ta tam 1 eksik değer var
    assert df["occupation"].isna().sum() == 1  # örnekte occupation'da tam 1 eksik değer var


def test_clean_drops_non_data_rows(raw_test):  # şart 3: çöp satır atılmış mı
    df = clean(raw_test)  # ham test verisini temizler
    assert len(df) == 2  # 3 satırdan biri açıklama satırıydı; geriye 2 gerçek kayıt kalmalı


def test_clean_target_values(raw_train, raw_test):  # şart 4: hedef sadece iki geçerli değer mi
    for raw in (raw_train, raw_test):  # hem eğitim hem test örneğini dener
        labels = set(clean(raw)[TARGET])  # temizlenmiş hedef sütundaki farklı değerleri toplar
        assert labels <= VALID_LABELS, f"Geçersiz etiketler: {labels - VALID_LABELS}"  # izinli kümenin dışında değer olmamalı


def test_clean_numeric_dtypes(raw_train):  # şart 5: sayısal sütunlar sayısal mı
    df = clean(raw_train)  # ham veriyi temizler
    for col in NUMERIC_COLS:  # her sayısal sütunu gezer
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} sayısal değil"  # sütun tipi sayısal olmalı


def test_clean_does_not_modify_input(raw_train):  # şart 6: girdi bozulmamış mı
    before = raw_train.copy()  # fonksiyondan önceki hâlin kopyasını saklar
    clean(raw_train)  # temizleme fonksiyonunu çalıştırır
    pd.testing.assert_frame_equal(raw_train, before)  # orijinal tablo hiç değişmemiş olmalı


def test_clean_keeps_problems_visible(raw_train):  # ⚠️ kuralı: eksik ve yinelenen satırlar silinmemiş mi
    df = clean(raw_train)  # ham veriyi temizler
    assert len(df) == 3  # 3 satırın hepsi kalmalı (eksik değerli satır da, kopya satır da)
    assert df.duplicated().sum() == 1  # yinelenen satır hâlâ görünür olmalı ki teşhis aracı yakalayabilsin
