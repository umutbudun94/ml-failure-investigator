"""Adult veri setini okuma ve temizleme.

İki aşama var:
  1. load_raw  -> dosyayı HİÇ dokunmadan okur (hazır, ben yazdım)
  2. clean     -> okunan ham veriyi kullanılabilir hâle getirir (GÖREV 1: sen yazacaksın)
"""  # modülün yapısını anlatan açıklama

from pathlib import Path  # dosya yollarıyla çalışmak için Path sınıfını içe aktarır

import pandas as pd  # tablo verisiyle çalışmak için pandas'ı "pd" kısaltmasıyla içe aktarır

from mlfi.config import COLUMN_NAMES, TEST_PATH, TRAIN_PATH  # sütun isimlerini ve dosya yollarını config modülünden alır


def load_raw(path: str | Path) -> pd.DataFrame:  # ham Adult dosyasını okuyup DataFrame döndüren fonksiyonu tanımlar
    """Ham dosyayı okur. Hiçbir temizlik yapmaz; her hücre metin (str) olarak gelir."""  # fonksiyonun ne yaptığını anlatır
    path = Path(path)  # gelen yolu (metin de olabilir) Path nesnesine çevirir
    if not path.exists():  # dosya diskte yoksa
        raise FileNotFoundError(  # anlaşılır bir hata fırlatır
            f"{path} bulunamadı. Veriyi README'deki 'Data' adımına göre data/raw/ içine koy."  # kullanıcıya ne yapacağını söyleyen hata mesajı
        )  # hata fırlatma ifadesinin sonu
    return pd.read_csv(  # CSV dosyasını okuyup sonucu döndürür
        path,  # okunacak dosya
        header=None,  # dosyanın ilk satırı başlık DEĞİL, veri (Adult'ta başlık satırı yok)
        names=COLUMN_NAMES,  # sütunlara config'teki isimleri verir
        dtype=str,  # her şeyi metin olarak okur; tip dönüşümüne clean() karar verecek
    )  # read_csv çağrısının sonu


def clean(df: pd.DataFrame) -> pd.DataFrame:  # ham DataFrame'i temizleyip yeni bir DataFrame döndürecek fonksiyon
    """GÖREV 1 — Bu fonksiyonu sen yazacaksın.

    Girdi : load_raw() çıktısı (her hücre metin).
    Çıktı : temizlenmiş YENİ bir DataFrame. Şartlar:

      1. Metin hücrelerinin başındaki/sonundaki boşluklar temizlenmiş olmalı.
      2. Veride "eksik" anlamına gelen işaret gerçek eksik değere (NaN) çevrilmiş olmalı.
      3. Veri olmayan satırlar (açıklama/çöp satırı) atılmış olmalı.
      4. Hedef sütun (income) SADECE "<=50K" veya ">50K" değerlerini içermeli.
      5. NUMERIC_COLS içindeki sütunlar sayısal tipte olmalı.
      6. Girdi olarak gelen df DEĞİŞMEMELİ (kopyası üzerinde çalış).

    ⚠️ YAPMAMAN gerekenler:
      - Eksik değerli satırları SİLME.
      - Yinelenen satırları SİLME.
      Neden? Bunlar sistemimizin TESPİT edip raporlaması gereken sorunlar.
      Temizlik aşamasında saklarsak teşhis aracımız onları hiç göremez.

    Doğru yazıp yazmadığını görmek için: terminalde `pytest` çalıştır.
    """  # görevin şartnamesi
    raise NotImplementedError(  # fonksiyon henüz yazılmadığı için bilinçli olarak hata fırlatır
        "Görev 1: clean() fonksiyonunu yaz. Şartlar fonksiyonun açıklamasında, kontrol için `pytest` çalıştır."  # ne yapılması gerektiğini söyleyen mesaj
    )  # hata fırlatma ifadesinin sonu


def load_adult() -> tuple[pd.DataFrame, pd.DataFrame]:  # eğitim ve test verisini temizlenmiş hâlde birlikte döndüren kısayol fonksiyonu
    """Eğitim ve test dosyalarını okuyup temizler. clean() yazılmadan çalışmaz."""  # fonksiyonun ne yaptığını anlatır
    train = clean(load_raw(TRAIN_PATH))  # eğitim dosyasını ham okuyup temizler
    test = clean(load_raw(TEST_PATH))  # test dosyasını ham okuyup temizler
    return train, test  # ikisini birlikte döndürür
