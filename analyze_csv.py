#!/usr/bin/env python3
"""CSE-CIC-IDS2018 (Thursday-01-03-2018) 분포 정찰.
라벨분포 / 크기·IAT 피처 히스토그램 / 정상vs공격 판별 top5 / 2D 결합분포.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "samples/cic2018/Thursday-01-03-2018.csv"

def load():
    df = pd.read_csv(CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # 숫자 변환 + inf/nan 정리
    for c in df.columns:
        if c not in ("Label","Timestamp"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf,-np.inf], np.nan)
    return df

def main():
    df = load()
    print(f"shape: {df.shape}")
    print("\n=== Label 분포 ===")
    print(df["Label"].value_counts())

    df["bin"] = np.where(df["Label"].str.strip().str.upper()=="BENIGN","Benign","Attack")
    nB = (df["bin"]=="Benign").sum(); nA=(df["bin"]=="Attack").sum()
    print(f"\nBenign={nB}  Attack={nA}")

    # ---- 크기/타이밍 관련 피처 선택 ----
    size_feats = ["Pkt Len Mean","Pkt Len Std","Fwd Pkt Len Mean","Bwd Pkt Len Mean","Pkt Size Avg"]
    time_feats = ["Flow IAT Mean","Flow IAT Std","Fwd IAT Mean","Flow Duration","Flow Pkts/s"]
    sel = [f for f in size_feats+time_feats if f in df.columns]

    # ---- Figure 1: 크기/IAT 피처 히스토그램 (Benign vs Attack) ----
    fig, ax = plt.subplots(2,5, figsize=(20,7))
    for i,f in enumerate(sel[:10]):
        a = ax[i//5, i%5]
        for grp,c in [("Benign","tab:blue"),("Attack","tab:red")]:
            x = df.loc[df["bin"]==grp, f].dropna().values
            x = x[np.isfinite(x)]
            if len(x)<5: continue
            # 로그 스케일(양수 헤비테일 대응)
            if np.nanmin(x)>=0 and np.nanmax(x)>0:
                x = np.log10(x+1.0)
                a.set_xlabel(f"log10({f}+1)")
            else:
                a.set_xlabel(f)
            a.hist(x, bins=60, alpha=0.5, density=True, color=c, label=grp)
        a.set_title(f, fontsize=9); a.legend(fontsize=7)
    fig.suptitle("CSE-CIC-IDS2018: size & timing feature distributions (Benign vs Attack)", fontsize=13)
    fig.tight_layout(); fig.savefig("figs/cic_feature_hists.png", dpi=110); plt.close(fig)

    # ---- 판별력 top5: 표준화된 평균차이 (Cohen's d 유사) ----
    num = df.select_dtypes(include=[np.number]).columns
    num = [c for c in num if c not in ("bin",)]
    rows=[]
    B = df[df["bin"]=="Benign"]; A = df[df["bin"]=="Attack"]
    for c in num:
        b = B[c].dropna(); a = A[c].dropna()
        if len(b)<10 or len(a)<10: continue
        sp = np.sqrt((b.var()+a.var())/2)+1e-12
        d = abs(b.mean()-a.mean())/sp
        if np.isfinite(d): rows.append((d,c))
    rows.sort(reverse=True)
    print("\n=== Benign vs Attack 판별력 top10 (|Cohen's d|) ===")
    for d,c in rows[:10]: print(f"  {d:6.2f}  {c}")
    top5 = [c for _,c in rows[:5]]

    # ---- Figure 2: 2D 결합분포 (판별력 1위 x 2위) ----
    fx, fy = top5[0], top5[1]
    sub = df[[fx,fy,"bin"]].replace([np.inf,-np.inf],np.nan).dropna()
    def tx(s):
        v = s.values.astype(float)
        return np.log10(v+1.0) if np.nanmin(v)>=0 else v
    fig, ax = plt.subplots(1,3, figsize=(18,5.5))
    # (a) 산점도
    for grp,c in [("Benign","tab:blue"),("Attack","tab:red")]:
        s = sub[sub["bin"]==grp].sample(min(4000,(sub["bin"]==grp).sum()), random_state=1)
        ax[0].scatter(tx(s[fx]), tx(s[fy]), s=4, alpha=0.3, c=c, label=grp)
    ax[0].set_title("Scatter (sampled)"); ax[0].set_xlabel(f"log10({fx})"); ax[0].set_ylabel(f"log10({fy})"); ax[0].legend()
    # (b)(c) 2D 히스토그램 density per class
    for k,(grp,cm) in enumerate([("Benign","Blues"),("Attack","Reds")]):
        s = sub[sub["bin"]==grp]
        ax[k+1].hist2d(tx(s[fx]), tx(s[fy]), bins=60, cmap=cm, density=True)
        ax[k+1].set_title(f"2D density: {grp}"); ax[k+1].set_xlabel(f"log10({fx})"); ax[k+1].set_ylabel(f"log10({fy})")
    fig.suptitle(f"CSE-CIC-IDS2018 joint distribution: {fx} × {fy}", fontsize=13)
    fig.tight_layout(); fig.savefig("figs/cic_joint2d.png", dpi=110); plt.close(fig)

    # ---- 복잡도 지표: 결합분포의 다봉성/상관 ----
    print(f"\n=== 결합분포 복잡도 ({fx} × {fy}) ===")
    for grp in ["Benign","Attack"]:
        s = sub[sub["bin"]==grp]
        r = np.corrcoef(tx(s[fx]), tx(s[fy]))[0,1]
        print(f"  {grp:7}: n={len(s):6}  corr={r:+.3f}")

if __name__=="__main__":
    import os; os.makedirs("figs", exist_ok=True)
    main()
    print("\nsaved: figs/cic_feature_hists.png, figs/cic_joint2d.png")
