#!/usr/bin/env python3
"""ISCX VPN-nonVPN (non-VPN 암호화 트래픽) 분포 정찰.
완전 암호화 → 신호는 패킷크기/IAT/방향에만. flow 단위 시퀀스 + 마진 + FFT + 2D결합 + 자기상관.
USTC-TFC2016 (FFT 평탄/고주파95%, 시간상관 거의 없음)과 비교.
"""
import dpkt, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

PCAPS = {
    "Chat (AIM)":    ("samples/iscx/aim_chat_3a.pcap",      "tab:green"),
    "Email":         ("samples/iscx/email1a.pcap",          "tab:orange"),
    "Audio (FB)":    ("samples/iscx/facebook_audio1a.pcap", "tab:purple"),
}
MAXPK = 60000

def flow_key(ip):
    try:
        a = (bytes(ip.src), ip.data.sport); b = (bytes(ip.dst), ip.data.dport)
        return (min(a,b), max(a,b)), (a < b)  # 정규화 5-tuple(2-way), fwd 여부
    except Exception:
        return None, True

def extract(path):
    lengths, times, dirs = [], [], []
    flows = defaultdict(int)
    first_src = None
    with open(path,"rb") as f:
        pcap = dpkt.pcap.Reader(f)
        for i,(ts,buf) in enumerate(pcap):
            if i>=MAXPK: break
            lengths.append(len(buf)); times.append(ts)
            d=0
            try:
                eth=dpkt.ethernet.Ethernet(buf); ip=eth.data
                if isinstance(ip,(dpkt.ip.IP,dpkt.ip6.IP6)):
                    src=bytes(ip.src)
                    if first_src is None: first_src=src
                    d = 1 if src==first_src else -1
                    fk,_=flow_key(ip)
                    if fk: flows[fk]+=1
            except Exception: pass
            dirs.append(d)
    L=np.array(lengths,float); T=np.array(times,float)
    iat=np.diff(T); iat=iat[iat>=0]
    pkts_per_flow=np.array(list(flows.values())) if flows else np.array([len(L)])
    return dict(lengths=L, iat=iat, dirs=np.array(dirs), n=len(L),
                nflows=len(flows), ppf=pkts_per_flow)

def hf_ratio(sig):
    sig=sig.astype(float); sig=sig-sig.mean(); n=len(sig)
    if n<16: return float("nan")
    spec=np.abs(np.fft.rfft(sig*np.hanning(n)))**2
    half=len(spec)//2
    return spec[half:].sum()/(spec[1:].sum()+1e-12)

def main():
    import os; os.makedirs("figs",exist_ok=True)
    data={k:extract(p) for k,(p,_) in PCAPS.items()}

    print("=== ISCX non-VPN 기본 통계 ===")
    for name,d in data.items():
        L,I=d["lengths"],d["iat"]
        print(f"{name:12} n={d['n']:6} flows={d['nflows']:4} pkts/flow med={np.median(d['ppf']):.0f} "
              f"| len mean={L.mean():.0f} skew={stats.skew(L):.2f} kurt={stats.kurtosis(L):.1f} "
              f"| IAT med={np.median(I)*1e3:.3f}ms skew={stats.skew(I):.1f} kurt={stats.kurtosis(I):.0f}")

    # Fig1: 마진 히스토그램
    fig,ax=plt.subplots(2,2,figsize=(13,8))
    for name,(p,c) in PCAPS.items():
        d=data[name]
        ax[0,0].hist(d["lengths"],bins=80,alpha=0.5,density=True,color=c,label=name)
        iat=d["iat"][d["iat"]>0]*1e3
        ax[0,1].hist(np.log10(iat+1e-9),bins=80,alpha=0.5,density=True,color=c,label=name)
        ax[1,0].plot(d["lengths"][:500],lw=0.6,color=c,alpha=0.8,label=name)
        ax[1,1].plot(np.cumsum(d["dirs"][:1000]),lw=0.9,color=c,label=name)
    ax[0,0].set_title("Packet length (density)");ax[0,0].set_xlabel("bytes");ax[0,0].legend()
    ax[0,1].set_title("IAT (log10 ms, density)");ax[0,1].set_xlabel("log10(IAT ms)");ax[0,1].legend()
    ax[1,0].set_title("Packet length seq (first 500)");ax[1,0].legend()
    ax[1,1].set_title("Direction cumulative walk (first 1000)");ax[1,1].legend()
    fig.suptitle("ISCX non-VPN: size / IAT / direction");fig.tight_layout()
    fig.savefig("figs/iscx_histograms.png",dpi=120);plt.close(fig)

    # Fig2: FFT
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    print("\n=== FFT 고주파 에너지 비율 ===")
    for name,(p,c) in PCAPS.items():
        d=data[name]
        for col,key in enumerate(["lengths","iat"]):
            sig=d[key].astype(float); sig=sig-sig.mean(); n=len(sig)
            if n<16: continue
            spec=np.abs(np.fft.rfft(sig*np.hanning(n))); spec/=spec.max()+1e-12
            freq=np.fft.rfftfreq(n,d=1.0)
            ax[col].semilogy(freq[1:],spec[1:]+1e-6,color=c,alpha=0.8,lw=0.7,label=name)
        print(f"  {name:12} len_HF={hf_ratio(d['lengths'])*100:5.1f}%  IAT_HF={hf_ratio(d['iat'])*100:5.1f}%")
    ax[0].set_title("Packet-size FFT");ax[0].set_xlabel("cycles/packet");ax[0].set_ylabel("norm mag (log)");ax[0].legend()
    ax[1].set_title("IAT FFT");ax[1].set_xlabel("cycles/packet");ax[1].legend()
    fig.suptitle("ISCX non-VPN: Fourier spectra");fig.tight_layout()
    fig.savefig("figs/iscx_fft.png",dpi=120);plt.close(fig)

    # Fig3: 2D 결합분포 (크기 × IAT), per pcap
    fig,ax=plt.subplots(1,3,figsize=(18,5.5))
    for j,(name,(p,c)) in enumerate(PCAPS.items()):
        d=data[name]
        L=d["lengths"][1:]; I=d["iat"]  # 길이 맞춤
        m=min(len(L),len(I)); L=L[:m]; I=I[:m]
        msk=I>0; lI=np.log10(I[msk]*1e3+1e-3); LL=L[msk]
        ax[j].hist2d(LL,lI,bins=70,cmap="viridis",density=True)
        ax[j].set_title(f"{name}: size x IAT");ax[j].set_xlabel("packet length (B)");ax[j].set_ylabel("log10(IAT ms)")
    fig.suptitle("ISCX non-VPN joint distribution (size x IAT)");fig.tight_layout()
    fig.savefig("figs/iscx_joint2d.png",dpi=110);plt.close(fig)

    # Fig4: 자기상관 (패킷크기 시퀀스)
    fig,ax=plt.subplots(1,1,figsize=(11,5))
    print("\n=== 자기상관 (패킷크기, lag=1..5) ===")
    for name,(p,c) in PCAPS.items():
        x=data[name]["lengths"].astype(float); x=x-x.mean(); n=len(x)
        if n<50: continue
        maxlag=min(200,n//4)
        ac=np.array([np.dot(x[:n-k],x[k:])/np.dot(x,x) for k in range(maxlag)])
        ax.plot(ac,color=c,lw=1.2,label=name)
        print(f"  {name:12} ACF[1..5]={np.round(ac[1:6],3)}")
    ax.axhline(0,color="k",lw=0.5);ax.set_title("Packet-size autocorrelation (periodicity from padding/crypto?)")
    ax.set_xlabel("lag (packets)");ax.set_ylabel("ACF");ax.legend()
    fig.tight_layout();fig.savefig("figs/iscx_autocorr.png",dpi=120);plt.close(fig)

    print("\nsaved: figs/iscx_histograms.png, iscx_fft.png, iscx_joint2d.png, iscx_autocorr.png")

if __name__=="__main__":
    main()
