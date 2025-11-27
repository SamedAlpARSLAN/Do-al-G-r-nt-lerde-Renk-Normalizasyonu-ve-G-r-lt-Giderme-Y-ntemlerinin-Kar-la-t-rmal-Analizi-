import argparse, os, random, glob, math
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

# ---------- Yardımcılar ----------
def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path, img_rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def to_float01(img):
    return img.astype(np.float32) / 255.0

def to_uint8(imgf):
    img = np.clip(imgf * 255.0, 0, 255).astype(np.uint8)
    return img

def psnr_ssim_rmse(ref_rgb, test_rgb):
    # skimage >= 0.19 için channel_axis=2
    ps = peak_signal_noise_ratio(ref_rgb, test_rgb, data_range=255)
    ss = structural_similarity(ref_rgb, test_rgb, channel_axis=2, data_range=255)
    mse = mean_squared_error(ref_rgb, test_rgb)
    rm = math.sqrt(mse)
    return ps, ss, rm, mse

# ---------- Gürültü ----------
def add_gaussian_noise(img_rgb, sigma=0.05):
    f = to_float01(img_rgb)
    noise = np.random.normal(0, sigma, f.shape).astype(np.float32)
    return to_uint8(np.clip(f + noise, 0, 1))

def add_salt_pepper(img_rgb, amount=0.02, salt_vs_pepper=0.5):
    out = img_rgb.copy()
    num = int(amount * out.shape[0] * out.shape[1])
    # salt
    coords = (np.random.randint(0, out.shape[0], num),
              np.random.randint(0, out.shape[1], num))
    out[coords[0], coords[1]] = [255, 255, 255]
    # pepper
    coords = (np.random.randint(0, out.shape[0], num),
              np.random.randint(0, out.shape[1], num))
    out[coords[0], coords[1]] = [0, 0, 0]
    return out

# ---------- Renk Normalizasyonu ----------
def reinhard_color_transfer(src_rgb, ref_rgb):
    # Lab uzayında kanal bazlı ort/std eşitleme (Reinhard)
    src = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        s_mean, s_std = src[:,:,c].mean(), src[:,:,c].std() + 1e-6
        r_mean, r_std = ref[:,:,c].mean(), ref[:,:,c].std() + 1e-6
        src[:,:,c] = ((src[:,:,c] - s_mean) * (r_std / s_std)) + r_mean
    out = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out

def gray_world_white_balance(img_rgb):
    # Ortalama kanal eşitleme
    img = to_float01(img_rgb)
    means = img.reshape(-1,3).mean(axis=0) + 1e-6
    scale = means.mean() / means
    out = img * scale
    return to_uint8(np.clip(out, 0, 1))

def histogram_match(src_rgb, ref_rgb):
    return match_histograms(src_rgb, ref_rgb, channel_axis=2).astype(np.uint8)

# ---------- Denoising ----------
def denoise_gaussian(img_rgb, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img_rgb, (ksize, ksize), sigma)

def denoise_median(img_rgb, ksize=3):
    return cv2.medianBlur(img_rgb, ksize)

def denoise_nl_means(img_rgb, h=10):
    # OpenCV fastNlMeans (renkli)
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, h, h, 7, 21)

# ---------- Çekirdek akış ----------
def collect_images(folder, exts=('*.jpg','*.jpeg','*.png','*.bmp')):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, e))
    return sorted(paths)

def ensure_ref_image(paths):
    # Liste içinden deterministik bir referans seç (0. indeks)
    if not paths: raise RuntimeError("Referans için görüntü bulunamadı.")
    return paths[0]

def plot_bars(means_csv, out_png):
    df = pd.read_csv(means_csv)
    # Basit bir özet bar grafiği: yöntem başına SSIM ortalaması
    subset = df[(df['noise']=='gaussian') & (df['denoise']!='none')]
    plt.figure()
    ax = subset.pivot_table(index='color_norm', columns='denoise', values='ssim', aggfunc='mean').plot(kind='bar')
    plt.ylabel("SSIM (ortalama)")
    plt.title("Renk Normalizasyonu + Denoise (SSIM)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/BSDS500/data/images",
                    help="BSDS500 images folder (…/data/BSDS500/data/images)")
    ap.add_argument("--setA", type=str, default="train")
    ap.add_argument("--setB", type=str, default="test")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--limit", type=int, default=50, help="Her veri setinden en fazla N görsel")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    dsA = os.path.join(args.root, args.setA)
    dsB = os.path.join(args.root, args.setB)

    A_paths = collect_images(dsA)
    B_paths = collect_images(dsB)
    if args.limit > 0:
        A_paths = A_paths[:args.limit]
        B_paths = B_paths[:args.limit]

    # Referans görüntüler (renk eşleştirme için)
    A_ref = imread_rgb(ensure_ref_image(A_paths)) if A_paths else None
    B_ref = imread_rgb(ensure_ref_image(B_paths)) if B_paths else None

    color_norms = {
        "none":        lambda x, r: x,
        "reinhard":    reinhard_color_transfer,
        "gray_world":  lambda x, r: gray_world_white_balance(x),
        "hist_match":  histogram_match
    }
    noises = {
        "none":        lambda x: x,
        "gaussian":    lambda x: add_gaussian_noise(x, sigma=0.05),
        "saltpepper":  lambda x: add_salt_pepper(x, amount=0.02)
    }
    denoisers = {
        "none":        lambda x: x,
        "gaussian":    denoise_gaussian,
        "median":      denoise_median,
        "nlmeans":     denoise_nl_means
    }

    rows = []
    def process_split(split_name, paths, ref_img):
        for p in tqdm(paths, desc=f"{split_name}"):
            base = os.path.splitext(os.path.basename(p))[0]
            img = imread_rgb(p)

            # “Renk norm → Gürültü ekle → Denoise” VE
            # “Gürültü ekle → Denoise” (renk norm yok) senaryolarını aynı döngüde kapsar
            for cn_name, cn_func in color_norms.items():
                cn_ref = ref_img if cn_name in ("reinhard","hist_match") else img
                img_cn = cn_func(img, cn_ref) if cn_name!="none" else img

                for nz_name, nz_func in noises.items():
                    img_noisy = nz_func(img_cn)
                    for dn_name, dn_func in denoisers.items():
                        img_out = dn_func(img_noisy)

                        # Değerlendirme stratejisi:
                        # - Gürültü eklendiyse: referans = orijinal img (gürültüden arındırma başarısı)
                        # - Gürültü yoksa: referans = orijinal img (renk normun yapısal sadakati)
                        ref = img
                        ps, ss, rm, mse = psnr_ssim_rmse(ref, img_out)

                        # Kaydet (birkaç örnek)
                        if random.random() < 0.02:
                            save_dir = os.path.join(args.out, "samples", split_name, cn_name, nz_name, dn_name)
                            imwrite_rgb(os.path.join(save_dir, f"{base}_orig.png"), img)
                            imwrite_rgb(os.path.join(save_dir, f"{base}_out.png"),  img_out)
                            imwrite_rgb(os.path.join(save_dir, f"{base}_noisy.png"), img_noisy)
                            if cn_name!="none": imwrite_rgb(os.path.join(save_dir, f"{base}_norm.png"), img_cn)

                        rows.append({
                            "split": split_name,
                            "image": base,
                            "color_norm": cn_name,
                            "noise": nz_name,
                            "denoise": dn_name,
                            "psnr": ps, "ssim": ss, "rmse": rm, "mse": mse
                        })

    if A_paths: process_split(args.setA, A_paths, A_ref)
    if B_paths: process_split(args.setB, B_paths, B_ref)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "metrics.csv")
    df.to_csv(csv_path, index=False)

    mean_df = (df
        .groupby(["split","color_norm","noise","denoise"], as_index=False)
        .agg(psnr=("psnr","mean"), ssim=("ssim","mean"), rmse=("rmse","mean"), mse=("mse","mean")))
    mean_csv = os.path.join(args.out, "metrics_mean.csv")
    mean_df.to_csv(mean_csv, index=False)

    # Basit özet grafiği
    plot_bars(mean_csv, os.path.join(args.out, "summary_ssim.png"))

    print(f"[OK] Kaydedildi: {csv_path}")
    print(f"[OK] Kaydedildi: {mean_csv}")
    print(f"[OK] Örnek görseller: results/samples/ ...")
    print(f"[OK] Grafik: results/summary_ssim.png")

if __name__ == "__main__":
    main()
