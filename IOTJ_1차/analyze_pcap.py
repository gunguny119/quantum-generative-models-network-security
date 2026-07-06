#!/usr/bin/env python3
"""USTC-TFC2016 PCAP 분포 정찰: 패킷크기/IAT/방향 시퀀스 추출 + 히스토그램 + FFT 스펙트럼.
양자 생성모델이 의미있을 만큼 분포가 '복잡/고주파'인지 약식 확인.
"""
import dpkt, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

PCAPS = {
    "Benign (Facetime)":  ("samples/Facetime.pcap", "tab:blue"),
    "Malware (Tinba)":    ("samples/Tinba.pcap",    "tab:red"),
}
MAXPK = 40000  # 맛보기 상한

def extract(path):
    """패킷길이, IAT(초), 방향(+1 fwd / -1 bwd) 시퀀스 추출. 방향은 첫 등장 IP 쌍 기준."""
    lengths, times, dirs = [], [], []
    first_src = None
    with open(path, "rb") as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except ValueError:
            return None
        for i, (ts, buf) in enumerate(pcap):
            if i >= MAXPK: break
            lengths.append(len(buf))
            times.append(ts)
            # 방향: IP src 기준. 첫 IP 패킷의 src를 forward로 간주
            d = 0
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    src = bytes(ip.src)
                    if first_src is None:
                        first_src = src
                    d = 1 if src == first_src else -1
            except Exception:
                d = 0
            dirs.append(d)
    lengths = np.array(lengths, float)
    times = np.array(times, float)
    iat = np.diff(times)
    iat = iat[iat >= 0]
    return dict(lengths=lengths, iat=iat, dirs=np.array(dirs), n=len(lengths))

def stats_line(name, d):
    L = d["lengths"]; I = d["iat"]
    from scipy import stats
    def hv(x):
        x = x[np.isfinite(x)]
        if len(x) < 3: return float("nan"), float("nan")
        return float(stats.skew(x)), float(stats.kurtosis(x))  # excess kurtosis
    ls, lk = hv(L); is_, ik = hv(I)
    return (f"{name}: n={d['n']}  | len mean={L.mean():.0f} std={L.std():.0f} "
            f"skew={ls:.2f} kurt={lk:.1f}  | IAT median={np.median(I)*1e3:.3f}ms "
            f"skew={is_:.2f} kurt={ik:.1f}")

def main():
    data = {k: extract(p) for k,(p,_) in PCAPS.items()}
    summary = []

    # ---- Figure 1: 히스토그램 (패킷크기 + IAT) ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for name,(p,c) in PCAPS.items():
        d = data[name]
        ax[0,0].hist(d["lengths"], bins=80, alpha=0.55, label=name, color=c, density=True)
        # IAT: 로그 스케일 (헤비테일 확인). 0 제거
        iat = d["iat"][d["iat"]>0]*1e3  # ms
        ax[0,1].hist(np.log10(iat+1e-9), bins=80, alpha=0.55, label=name, color=c, density=True)
        # 패킷 크기 시퀀스 (앞 500개) — 시간축 모양
        ax[1,0].plot(d["lengths"][:500], lw=0.6, color=c, alpha=0.8, label=name)
        # 방향 누적합 (랜덤워크 모양)
        ax[1,1].plot(np.cumsum(d["dirs"][:1000]), lw=0.9, color=c, label=name)
        summary.append(stats_line(name, d))
    ax[0,0].set_title("Packet length distribution (density)"); ax[0,0].set_xlabel("bytes"); ax[0,0].legend()
    ax[0,1].set_title("Inter-arrival time (log10 ms, density)"); ax[0,1].set_xlabel("log10(IAT ms)"); ax[0,1].legend()
    ax[1,0].set_title("Packet length sequence (first 500)"); ax[1,0].set_xlabel("packet idx"); ax[1,0].set_ylabel("bytes"); ax[1,0].legend()
    ax[1,1].set_title("Direction cumulative walk (first 1000)"); ax[1,1].set_xlabel("packet idx"); ax[1,1].legend()
    fig.suptitle("USTC-TFC2016: packet-size / IAT / direction distributions", fontsize=13)
    fig.tight_layout(); fig.savefig("figs/ustc_histograms.png", dpi=120); plt.close(fig)

    # ---- Figure 2: FFT 스펙트럼 (패킷크기 시퀀스 + IAT 시퀀스) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name,(p,c) in PCAPS.items():
        d = data[name]
        for col, (sig_key, title) in enumerate([("lengths","Packet-size sequence FFT"),
                                                 ("iat","IAT sequence FFT")]):
            sig = d[sig_key].astype(float)
            sig = sig - sig.mean()             # DC 제거
            n = len(sig)
            if n < 16: continue
            # 윈도우 적용
            w = np.hanning(n)
            spec = np.abs(np.fft.rfft(sig*w))
            freq = np.fft.rfftfreq(n, d=1.0)   # 정규화 주파수 (cycles/packet)
            # 파워 정규화
            spec = spec / (spec.max()+1e-12)
            ax[col].semilogy(freq[1:], spec[1:]+1e-6, color=c, alpha=0.8, lw=0.8, label=name)
            ax[col].set_title(title); ax[col].set_xlabel("normalized freq (cycles/packet)")
            ax[col].set_ylabel("normalized magnitude (log)"); ax[col].legend()
    fig.suptitle("USTC-TFC2016: Fourier spectra — low-freq(smooth) vs high-freq(complex)?", fontsize=13)
    fig.tight_layout(); fig.savefig("figs/ustc_fft.png", dpi=120); plt.close(fig)

    # ---- 고주파 에너지 비율 계산 (객관 지표) ----
    print("\n=== USTC-TFC2016 분포 요약 ===")
    for s in summary: print(" ", s)
    print("\n--- FFT 고주파 에너지 비율 (상위 50% 주파수대 파워 / 전체) ---")
    for name in PCAPS:
        d = data[name]
        for sig_key in ("lengths","iat"):
            sig = d[sig_key].astype(float); sig = sig-sig.mean()
            n=len(sig)
            if n<16: continue
            spec = np.abs(np.fft.rfft(sig*np.hanning(n)))**2
            half = len(spec)//2
            hf = spec[half:].sum()/(spec[1:].sum()+1e-12)
            print(f"  {name:20} {sig_key:8}: 고주파에너지={hf*100:5.1f}%")

if __name__ == "__main__":
    import os; os.makedirs("figs", exist_ok=True)
    main()
    print("\nsaved: figs/ustc_histograms.png, figs/ustc_fft.png")
