$readmeContent = @'
# Doğal Görüntülerde Renk Normalizasyonu ve Gürültü Giderme Analizi

**Öğrenci:** Samed Alp Arslan (220205012)  
**Ders:** Sayısal Görüntü İşleme (Digital Image Processing)

## 📄 Proje Özeti
Bu çalışma, **BSDS500** doğal görüntü veri kümesi üzerinde farklı **renk normalizasyonu** ve **gürültü giderme (denoise)** yöntemlerinin performansını karşılaştırmalı olarak analiz eder.

Amaç; renk sapmalarını ve gürültüyü giderirken görüntünün yapısal bütünlüğünü (SSIM) en iyi koruyan kombinasyonu belirlemektir.

---

## 🛠️ Kullanılan Yöntemler

### 1. Gürültü Modelleri
Deneylerde görüntülere yapay olarak şu gürültüler eklenmiştir:
- **Gauss Gürültüsü:** $\sigma=0.05$
- **Tuz-Biber (Salt & Pepper):** Yoğunluk $\approx\%2$

### 2. Renk Normalizasyonu Teknikleri
- **Reinhard:** Lab renk uzayında istatistiksel eşleştirme.
- **Gray-World:** Kanal ortalamalarını eşitleyerek beyaz dengeleme.
- **Histogram Eşleştirme:** Referans görüntü histogramına uydurma.
- **None:** Normalizasyon uygulanmayan kontrol grubu.

### 3. Gürültü Giderme (Denoise) Filtreleri
- **Gaussian Blur:** Gauss gürültüsü için (ksize=5).
- **Median Filtre:** Tuz-biber gürültüsü için (ksize=3).
- **Non-Local Means (NLM):** Dokusal detayları koruyan gelişmiş filtreleme.

---

## 📊 Değerlendirme Metrikleri
Başarım ölçümü için orijinal görüntüler referans alınarak şu metrikler kullanılmıştır:
1. **PSNR** (Peak Signal-to-Noise Ratio)
2. **SSIM** (Structural Similarity Index)
3. **RMSE** (Root Mean Square Error)

---

## 📈 Bulgular ve Sonuçlar

Deney sonuçlarına göre öne çıkan bulgular:

### Genel Başarım (SSIM)
| Senaryo | Denoise Yöntemi | SSIM Başarısı | Yorum |
|---------|-----------------|---------------|-------|
| **Gray-World** | Non-Local Means | ⭐ Yüksek | Yapısal benzerliği en iyi koruyan kombinasyon. |
| **None** | Gaussian Blur | 🟢 Orta | Gauss gürültüsünde etkili ancak detay kaybı var. |
| **Reinhard** | (Tümü) | 🔴 Düşük | Renk istatistiklerini agresif değiştirdiği için SSIM düşmektedir. |

### Örnek Sonuçlar (Görsel)
![Sonuç Örneği](results/summary_ssim.png)
*Şekil: Farklı yöntemlerin ortalama SSIM karşılaştırması.*

### Sonuç
Doğal görüntülerde **Gray-World + NLM** veya **None + NLM** kombinasyonları en tutarlı sonuçları vermiştir. Reinhard ve Histogram eşleştirme gibi yöntemler, doğal sahnelerin renk karakteristiğini bozabildiği için yapısal benzerlik skorlarını düşürmüştür.

---

## 🚀 Kurulum

Gerekli kütüphaneleri yüklemek için:

pip install -r requirements.txt
Projeyi çalıştırmak için:
python main.py

***
***
***
# Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)

## Overview

The goal of this work is to provide an empirical basis for research on image
segmentation and boundary detection. In order to promote scientific progress
in the study of visual grouping, we provide the following resources:

- A large dataset of natural images that have been manually segmented. The
  human annotations serve as ground truth for learning grouping cues as well
  as a benchmark for comparing different segmentation and boundary detection
  algorithms.

- The most recent algorithms our group has developed for contour detection and
  image segmentation.

- Performance evaluation of the leading computational approaches to grouping.

This is a mirror of the January 2013 update.

If you use the resources in this page, please cite the paper:

Contour Detection and Hierarchical Image Segmentation
P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.
IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.
[PDF](http://web.archive.org/web/20160306133802/http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf)
[BiBTeX](http://web.archive.org/web/20160306133802/http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2011.bib)

For more information, please [read the original dataset
description](http://web.archive.org/web/20160306133802/http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500)
