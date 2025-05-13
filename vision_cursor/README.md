# VisionCursor

VisionCursor, üst uzuvlarını kullanamayan bireylerin bilgisayarı gözleri ve sesleriyle kullanabilmelerini sağlayan bir Python masaüstü uygulamasıdır.

## Özellikler

- **Göz Takibi ile İmleç Kontrolü**: Bilgisayarın webcam'i kullanılarak göz hareketleriyle imlecin kontrol edilmesi
- **Sesli Yazı Yazma**: Konuşmaların gerçek zamanlı olarak metne dönüştürülmesi
- **Kullanıcı Dostu Arayüz**: Dokunmatik olmayan kontrole uygun tasarlanmış arayüz
- **Paralel İşlem**: Göz takibi ve ses tanıma özelliklerinin eş zamanlı çalışması

## Kurulum

1. Python 3.8 veya üst sürümü kurulu olmalıdır.
2. Aşağıdaki komutu kullanarak gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

3. Dlib kurulumu için C++ derleyicisi gereklidir.

## Kullanım

Uygulamayı başlatmak için:

```bash
python main.py
```

### Temel Kullanım:

1. Uygulama başladığında, kameranızın doğru çalıştığından emin olun.
2. Göz takibi başlat düğmesine tıklayın veya "başlat" komutunu sesli olarak söyleyin.
3. Gözlerinizle imleci kontrol edin, belirli bir noktada 2 saniye sabitlediğinizde tıklama gerçekleşecektir.
4. Ses tanıma özelliğini kullanmak için, ilgili düğmeye tıklayın veya "ses tanıma başlat" komutunu söyleyin.

## Sistem Gereksinimleri

- Windows 10/11, macOS 10.15+ veya Linux
- Kamera (dahili veya harici)
- Mikrofon
- En az 4GB RAM 