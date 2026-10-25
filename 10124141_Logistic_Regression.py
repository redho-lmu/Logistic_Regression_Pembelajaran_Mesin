"""
=====================================================================
TUGAS PEMBELAJARAN MESIN: LOGISTIC REGRESSION
Dataset : Breast Cancer Wisconsin (Diagnostic)
Kelas   : M (Malignant/ganas) = 1  |  B (Benign/jinak) = 0
=====================================================================
Program ini berisi dua bagian sesuai tugas:

  BAGIAN A : 5 data latih + 1 data uji
             - tampilkan 5 data latih
             - pelatihan langkah-per-langkah (pseudo-code) 2 epoch
             - klasifikasi 1 data uji
  BAGIAN B : seluruh data, pembagian 80% latih : 20% uji
             - logistic regression (implementasi sendiri, gradient descent)
             - evaluasi: confusion matrix, akurasi, presisi, recall, F1
             - pembanding: scikit-learn LogisticRegression
"""

import os
import sys
import numpy as np
import pandas as pd

np.set_printoptions(precision=6, suppress=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# ---------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"

# --- Bagian A ---
FITUR_A = ["radius_mean", "texture_mean"]        # 2 fitur (f1, f2) agar mudah dihitung manual
IDX_LATIH_A = [0, 1, 2, 19, 20]                   # 3 data M + 2 data B (indeks baris pada data.csv)
IDX_UJI_A = 4                                     # 1 data uji (baris ke-4 pada data.csv, kelas M)
W_AWAL_A = np.array([0.1, 0.1])                   # bobot awal (ditentukan peneliti)
B_AWAL_A = 0.0                                    # bias awal
ETA_A = 0.5                                       # learning rate
EPOCH_A = 2                                       # jumlah epoch

# --- Bagian B ---
UKURAN_UJI_B = 0.20                               # 80% : 20%
SEED_B = 42
ETA_B = 0.1
EPOCH_B = 2000


# ---------------------------------------------------------------------
# FUNGSI-FUNGSI INTI (pseudo-code)
# ---------------------------------------------------------------------
def sigmoid(z):
    """sigma(z) = 1 / (1 + e^-z)  (versi stabil secara numerik)."""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def hitung_z(X, w, b):
    """z = sum(w_i x_i) + b"""
    return X @ w + b


def loss_cross_entropy(y, s, eps=1e-12):
    """L_CE = -(y ln s + (1-y) ln(1-s)) untuk tiap sampel."""
    s = np.clip(s, eps, 1 - eps)
    return -(y * np.log(s) + (1 - y) * np.log(1 - s))


def gradien(X, y, s):
    """dJ/dw = (1/N) sum (s-y) x ;  dJ/db = (1/N) sum (s-y)"""
    N = len(y)
    err = s - y
    dw = (err[:, None] * X).sum(axis=0) / N
    db = err.sum() / N
    return dw, db


def latih_logistic(X, y, w, b, eta, epoch, verbose=False, nama_fitur=None):
    """Gradient descent batch. Mengembalikan (w, b, riwayat_J)."""
    w = w.astype(float).copy()
    b = float(b)
    riwayat = []
    for ep in range(1, epoch + 1):
        z = hitung_z(X, w, b)
        s = sigmoid(z)
        L = loss_cross_entropy(y, s)
        J = L.mean()
        riwayat.append(J)
        dw, db = gradien(X, y, s)

        if verbose:
            cetak_epoch(ep, X, y, w, b, z, s, L, J, dw, db, eta, nama_fitur)

        w = w - eta * dw
        b = b - eta * db
    return w, b, riwayat


def prediksi(X, w, b, ambang=0.5):
    p = sigmoid(hitung_z(X, w, b))
    return (p > ambang).astype(int), p


def standarisasi_fit(X):
    """Hitung mean & std (populasi, ddof=0) dari DATA LATIH saja."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return mu, sd


def metrik(y_true, y_pred):
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    total = TP + TN + FP + FN
    akurasi = (TP + TN) / total
    presisi = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * presisi * recall / (presisi + recall) if (presisi + recall) else 0.0
    return dict(TP=TP, TN=TN, FP=FP, FN=FN, akurasi=akurasi,
                presisi=presisi, recall=recall, f1=f1)


def cetak_metrik(nama, m):
    print(f"\n[{nama}]")
    print("Confusion matrix (baris = aktual, kolom = prediksi)")
    print("                 Pred B(0)   Pred M(1)")
    print(f"  Aktual B(0)   {m['TN']:>9d}   {m['FP']:>9d}")
    print(f"  Aktual M(1)   {m['FN']:>9d}   {m['TP']:>9d}")
    print(f"  Akurasi : {m['akurasi']:.4f}")
    print(f"  Presisi : {m['presisi']:.4f}")
    print(f"  Recall  : {m['recall']:.4f}")
    print(f"  F1-score: {m['f1']:.4f}")


def garis(judul=""):
    print("\n" + "=" * 100)
    if judul:
        print(judul)
        print("=" * 100)


# ---------------------------------------------------------------------
# CETAK TABEL PER EPOCH (langkah demi langkah)
# ---------------------------------------------------------------------
def cetak_epoch(ep, X, y, w, b, z, s, L, J, dw, db, eta, nama_fitur):
    f1, f2 = nama_fitur if nama_fitur else ("f1", "f2")
    err = s - y
    g1 = err * X[:, 0]
    g2 = err * X[:, 1]
    N = len(y)

    print(f"\n--- EPOCH {ep} ---")
    print(f"Parameter awal epoch : w = ({w[0]:.6f}, {w[1]:.6f}),  b = {b:.6f},  eta = {eta}")
    tabel = pd.DataFrame({
        "i": np.arange(1, N + 1),
        "f1": X[:, 0], "f2": X[:, 1],
        "z": z, "sigma": s, "y": y.astype(int),
        "L_CE": L, "(s-y)": err, "(s-y)*f1": g1, "(s-y)*f2": g2,
    })
    print(tabel.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"Total L_CE = {L.sum():.6f}   ->   J = (1/N) * total = {J:.6f}")
    print(f"Sum (s-y)      = {err.sum():.6f}")
    print(f"Sum (s-y)*f1   = {g1.sum():.6f}")
    print(f"Sum (s-y)*f2   = {g2.sum():.6f}")
    print(f"dJ/dw1 = {g1.sum():.6f}/{N} = {dw[0]:.6f}")
    print(f"dJ/dw2 = {g2.sum():.6f}/{N} = {dw[1]:.6f}")
    print(f"dJ/db  = {err.sum():.6f}/{N} = {db:.6f}")
    w_baru = w - eta * dw
    b_baru = b - eta * db
    print(f"w1 baru = {w[0]:.6f} - {eta}*({dw[0]:.6f}) = {w_baru[0]:.6f}")
    print(f"w2 baru = {w[1]:.6f} - {eta}*({dw[1]:.6f}) = {w_baru[1]:.6f}")
    print(f"b  baru = {b:.6f} - {eta}*({db:.6f}) = {b_baru:.6f}")


# ---------------------------------------------------------------------
# MUAT & PRAPROSES DATA
# ---------------------------------------------------------------------
def muat_data(path):
    df = pd.read_csv(path)
    # kolom 'Unnamed: 32' kosong (artefak koma di akhir baris) dan 'id' bukan fitur
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["y"] = (df["diagnosis"] == "M").astype(int)     # M = 1 (positif), B = 0
    return df


# =====================================================================
# BAGIAN A : 5 DATA LATIH + 1 DATA UJI
# =====================================================================
def bagian_a(df):
    garis("BAGIAN A  |  5 DATA LATIH + 1 DATA UJI")

    # ---- Langkah 1: tampilkan 5 data latih -------------------------
    latih = df.loc[IDX_LATIH_A, ["id", "diagnosis", "y"] + FITUR_A]
    uji = df.loc[[IDX_UJI_A], ["id", "diagnosis", "y"] + FITUR_A]
    print("\nLANGKAH 1 - 5 DATA LATIH (data asli, 2 kelas: M dan B)")
    print(latih.to_string())
    print("\nDATA UJI (1 data)")
    print(uji.to_string())
    print("\nCatatan: y = 1 untuk M (Malignant), y = 0 untuk B (Benign). "
          f"Fitur: f1 = {FITUR_A[0]}, f2 = {FITUR_A[1]}")

    X_raw = latih[FITUR_A].to_numpy(float)
    y = latih["y"].to_numpy(float)
    X_uji_raw = uji[FITUR_A].to_numpy(float)

    # ---- Praproses: standarisasi (statistik dari DATA LATIH) -------
    mu, sd = standarisasi_fit(X_raw)
    X = (X_raw - mu) / sd
    X_uji = (X_uji_raw - mu) / sd
    print("\nPRAPROSES - Standarisasi z-score:  x' = (x - mean) / std   (std populasi, dari 5 data latih)")
    print(f"  mean = {mu},  std = {sd}")
    tab = pd.DataFrame({"i": range(1, 6), "diagnosis": latih["diagnosis"].values,
                        "f1 (std)": X[:, 0], "f2 (std)": X[:, 1], "y": y.astype(int)})
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # ---- Langkah 2: pelatihan 2 epoch ------------------------------
    print("\nLANGKAH 2 - PELATIHAN LOGISTIC REGRESSION (gradient descent batch), "
          f"{EPOCH_A} epoch")
    print(f"Inisialisasi: w = ({W_AWAL_A[0]}, {W_AWAL_A[1]}), b = {B_AWAL_A}, eta = {ETA_A}")
    w, b, riwayat = latih_logistic(X, y, W_AWAL_A, B_AWAL_A, ETA_A, EPOCH_A,
                                   verbose=True, nama_fitur=FITUR_A)
    print(f"\nHasil setelah {EPOCH_A} epoch: w = ({w[0]:.6f}, {w[1]:.6f}), b = {b:.6f}")

    # ---- Langkah 3: klasifikasi data uji ---------------------------
    print("\nLANGKAH 3 - KLASIFIKASI DATA UJI")
    print(f"Data uji asli      : {X_uji_raw[0]}")
    print(f"Data uji distandarisasi: {X_uji[0]}")
    z_uji = hitung_z(X_uji, w, b)
    p1 = float(sigmoid(z_uji)[0])
    kelas = "M (ganas, 1)" if p1 > 0.5 else "B (jinak, 0)"
    print(f"z = {w[0]:.6f}*({X_uji[0,0]:.6f}) + {w[1]:.6f}*({X_uji[0,1]:.6f}) + {b:.6f} = {z_uji[0]:.6f}")
    print(f"P(y=1|x) = sigma(z) = {p1:.6f}")
    print(f"P(y=0|x) = 1 - sigma(z) = {1 - p1:.6f}")
    print(f"Aturan: sigma(z) > 0.5 -> kelas 1 (M), selain itu kelas 0 (B)")
    print(f"HASIL KLASIFIKASI : {kelas}   |   Kelas sebenarnya: {uji['diagnosis'].iloc[0]}")

    # ---- Langkah 4a: versi program (bukti kesesuaian) ---------------
    print("\nLANGKAH 4a - PROGRAM: klasifikasi 5 data latih + 1 data uji")
    print("(a) Implementasi sendiri, 2 epoch (harus sama dengan hitungan manual di atas)")
    print(f"    w = {w}, b = {b:.6f}, P(y=1|x_uji) = {p1:.6f}, prediksi = {int(p1 > 0.5)}")

    w_k, b_k, riw = latih_logistic(X, y, W_AWAL_A, B_AWAL_A, ETA_A, 1000)
    pred_latih, _ = prediksi(X, w_k, b_k)
    pred_uji, p_uji_k = prediksi(X_uji, w_k, b_k)
    print("(b) Implementasi sendiri, dilatih sampai konvergen (1000 epoch)")
    print(f"    w = {w_k}, b = {b_k:.6f}, J akhir = {riw[-1]:.6f}")
    print(f"    Akurasi data latih = {(pred_latih == y).mean():.2f}")
    print(f"    P(y=1|x_uji) = {float(p_uji_k[0]):.6f}, prediksi = {int(pred_uji[0])} "
          f"({'M' if pred_uji[0] == 1 else 'B'}), aktual = {uji['diagnosis'].iloc[0]}")

    try:
        from sklearn.linear_model import LogisticRegression
        sk = LogisticRegression(C=1e6, max_iter=10000)   # C besar ~ tanpa regularisasi
        sk.fit(X, y)
        p_sk = sk.predict_proba(X_uji)[0, 1]
        print("(c) Pembanding scikit-learn LogisticRegression (tanpa regularisasi berarti)")
        print(f"    w = {sk.coef_[0]}, b = {sk.intercept_[0]:.6f}")
        print(f"    P(y=1|x_uji) = {p_sk:.6f}, prediksi = {int(sk.predict(X_uji)[0])}")
    except ImportError:
        print("(c) scikit-learn tidak terpasang, pembanding dilewati.")


# =====================================================================
# BAGIAN B : SEMUA DATA, 80% LATIH : 20% UJI
# =====================================================================
def split_stratified(y, test_size, seed):
    """Pembagian acak berstrata: proporsi M/B pada latih & uji dijaga sama."""
    rng = np.random.default_rng(seed)
    idx_uji = []
    for kelas in np.unique(y):
        idx = np.where(y == kelas)[0]
        rng.shuffle(idx)
        n_uji = int(round(len(idx) * test_size))
        idx_uji.extend(idx[:n_uji])
    idx_uji = np.array(sorted(idx_uji))
    idx_latih = np.array(sorted(set(range(len(y))) - set(idx_uji)))
    return idx_latih, idx_uji


def bagian_b(df):
    garis("BAGIAN B  |  SEMUA DATA, PEMBAGIAN 80% LATIH : 20% UJI")

    fitur = [c for c in df.columns if c not in ("id", "diagnosis", "y")]
    X_all = df[fitur].to_numpy(float)
    y_all = df["y"].to_numpy(float)

    print(f"\nJumlah data   : {len(df)}  (B = {(y_all == 0).sum()}, M = {(y_all == 1).sum()})")
    print(f"Jumlah fitur  : {len(fitur)}")
    print(f"Missing value : {int(df[fitur].isna().sum().sum())}")

    idx_latih, idx_uji = split_stratified(y_all, UKURAN_UJI_B, SEED_B)
    X_tr, y_tr = X_all[idx_latih], y_all[idx_latih]
    X_te, y_te = X_all[idx_uji], y_all[idx_uji]
    print(f"Data latih    : {len(idx_latih)} ({len(idx_latih)/len(df):.0%})  "
          f"[B = {(y_tr == 0).sum()}, M = {(y_tr == 1).sum()}]")
    print(f"Data uji      : {len(idx_uji)} ({len(idx_uji)/len(df):.0%})  "
          f"[B = {(y_te == 0).sum()}, M = {(y_te == 1).sum()}]")

    # standarisasi: fit HANYA pada data latih (mencegah data leakage)
    mu, sd = standarisasi_fit(X_tr)
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd

    # latih (implementasi sendiri)
    w, b, riw = latih_logistic(X_tr_s, y_tr, np.zeros(X_tr_s.shape[1]), 0.0, ETA_B, EPOCH_B)
    print(f"\nPelatihan (implementasi sendiri): eta = {ETA_B}, epoch = {EPOCH_B}")
    for ep in (1, 10, 100, 500, 1000, EPOCH_B):
        print(f"  Epoch {ep:>5d}  J = {riw[ep - 1]:.6f}")

    pred_tr, _ = prediksi(X_tr_s, w, b)
    pred_te, p_te = prediksi(X_te_s, w, b)
    cetak_metrik("Implementasi sendiri - DATA LATIH (80%)", metrik(y_tr, pred_tr))
    m_te = metrik(y_te, pred_te)
    cetak_metrik("Implementasi sendiri - DATA UJI (20%)", m_te)

    # 5 koefisien terbesar (kontribusi terhadap kelas M)
    urut = np.argsort(-np.abs(w))[:5]
    print("\n5 fitur dengan |bobot| terbesar (data terstandarisasi):")
    for i in urut:
        print(f"  {fitur[i]:<26s} w = {w[i]:+.4f}   odds ratio = exp(w) = {np.exp(w[i]):.3f}")

    # pembanding scikit-learn
    try:
        from sklearn.linear_model import LogisticRegression
        sk = LogisticRegression(max_iter=10000)
        sk.fit(X_tr_s, y_tr)
        pred_sk = sk.predict(X_te_s)
        cetak_metrik("Pembanding scikit-learn - DATA UJI (20%)", metrik(y_te, pred_sk))
    except ImportError:
        print("\nscikit-learn tidak terpasang, pembanding dilewati.")


# ---------------------------------------------------------------------
# MENU UTAMA
# ---------------------------------------------------------------------
def bersihkan_layar():
    os.system("cls" if os.name == "nt" else "clear")


def tampilkan_menu(n_data):
    print("=" * 60)
    print("   TUGAS PEMBELAJARAN MESIN")
    print("   PROGRAM LOGISTIC REGRESSION - BREAST CANCER WISCONSIN")
    print("=" * 60)
    print(f"   Data dimuat : {CSV_PATH}  ({n_data} baris)")
    print("-" * 60)
    print("\n   [1]  Bagian A : 5 data latih + 1 data uji (2 epoch)")
    print("   [2]  Bagian B : Semua data, pembagian 80% : 20%")
    print("   [0]  Keluar")
    print(" ")
    print("=" * 60)


def main():
    try:
        data = muat_data(CSV_PATH)
    except FileNotFoundError:
        print(f"File '{CSV_PATH}' tidak ditemukan.")
        print("Letakkan data.csv di folder yang sama, atau jalankan:")
        print("    python logistic_regression_breast_cancer.py path/ke/data.csv")
        return

    while True:
        bersihkan_layar()
        tampilkan_menu(len(data))
        try:
            pilihan = input("Pilih menu (0/1/2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if pilihan == "1":
            bersihkan_layar()
            bagian_a(data)
        elif pilihan == "2":
            bersihkan_layar()
            bagian_b(data)
        elif pilihan == "0":
            print("Terima kasih. Program selesai.")
            break
        else:
            input("Pilihan tidak valid. Tekan Enter untuk kembali ke menu...")
            continue

        garis("SELESAI")
        try:
            input("Tekan Enter untuk kembali ke menu...")
        except (EOFError, KeyboardInterrupt):
            print()
            break


if __name__ == "__main__":
    main()
