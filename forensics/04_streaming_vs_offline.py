"""
Forensic #4: Streaming vs Offline Feature Comparison
Tests W_ROLLING mismatch (research=1000 vs trader=5000)
"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from math import sqrt
from datetime import timedelta
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent
TICK_DIR = BASE_DIR / "Data" / "Raw" / "Ticks"
OUTPUT_DIR = BASE_DIR / "outputs"
SNAPSHOT_DIR = OUTPUT_DIR / "features" / "snapshots"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class RZS:
    def __init__(self, w=1000):
        self.w=w; self.d=deque(maxlen=w); self.mean=0.0; self.M2=0.0
    def update(self, x):
        N=len(self.d)
        if N==self.w:
            xo=self.d[0]; self.d.append(x)
            mn=self.mean+(x-xo)/N; self.M2+=((x-xo)*((x-mn)+(xo-self.mean))); self.mean=mn
            if self.M2<0: self.M2=0.0
            s=sqrt(self.M2/(N-1)) if N>1 else 0.0
            return 0.0 if s<1e-12 else (x-self.mean)/s
        else:
            self.d.append(x); N=len(self.d)
            if N<30: return 0.0
            a=list(self.d); self.mean=sum(a)/N; self.M2=sum((v-self.mean)**2 for v in a)
            s=sqrt(self.M2/(N-1)) if N>1 else 0.0
            return 0.0 if s<1e-12 else (x-self.mean)/s

def load_ticks(date):
    p=TICK_DIR/str(date.year)/f"{date.month:02d}"/f"{date.day:02d}.parquet"
    if not p.exists(): return pd.DataFrame(columns=["timestamp","bid","bid_vol","ask","ask_vol"])
    df=pd.read_parquet(p)
    if df["timestamp"].dt.tz is None: df["timestamp"]=df["timestamp"].dt.tz_localize("UTC")
    return df

def ofi(bk,bkm,ak,akm,bvk,bvkm,avk,avkm):
    db=bk-bkm; da=ak-akm
    return ((1 if db>=0 else 0)*bvk-(1 if db<=0 else 0)*bvkm-(1 if da<=0 else 0)*avk+(1 if da>=0 else 0)*avkm)

def main():
    print("="*60+"\n FORENSIC #4: Streaming vs Offline Feature Comparison\n"+"="*60)
    labels=pd.read_parquet(OUTPUT_DIR/"labels.parquet")
    labels["date"]=pd.to_datetime(labels["date"],utc=True)
    test_bricks=labels.tail(300).reset_index(drop=True)
    feat_names=["z_OFI","z_Depth","z_Susc","z_Vel","z_Spread"]
    results={}
    for ws in [1000,5000]:
        print(f"\n Testing W_ROLLING={ws}")
        zs=[RZS(ws) for _ in range(5)]
        # Warmup
        ws_date=test_bricks.iloc[0]["date"]-timedelta(days=30)
        we_date=test_bricks.iloc[0]["date"]
        pb,pa,pbv,pav,pts=None,None,None,None,None
        wc=0; cd=ws_date.normalize()
        while cd<we_date:
            t=load_ticks(cd)
            if len(t)>0:
                for i in range(len(t)):
                    b,a,bv,av=t.iloc[i]["bid"],t.iloc[i]["ask"],t.iloc[i]["bid_vol"],t.iloc[i]["ask_vol"]
                    ts=t.iloc[i]["timestamp"].value/1e6
                    if pb is not None:
                        o=ofi(b,pb,a,pa,bv,pbv,av,pav); d=bv+av; su=o/(d+1e-8)
                        v=1.0/((ts-pts)+1e-3); sp=a-b
                        for j,val in enumerate([o,d,su,v,sp]): zs[j].update(val)
                        wc+=1
                    pb,pa,pbv,pav,pts=b,a,bv,av,ts
            cd+=timedelta(days=1)
        print(f"  Warmup: {wc:,} ticks")
        svecs,ovecs=[],[]
        nc=0
        for bi in range(len(test_bricks)):
            row=test_bricks.iloc[bi]; bid=int(row["brick_id"]); bc=row["date"]
            sp=SNAPSHOT_DIR/f"snapshot_{bid}.npy"
            if not sp.exists(): continue
            osn=np.load(sp)
            li=-1
            for r in range(99,-1,-1):
                if np.any(osn[r]!=0): li=r; break
            if li<0: continue
            ov=osn[li,:5]
            pbc=test_bricks.iloc[bi-1]["date"] if bi>0 else bc-timedelta(hours=1)
            tf=[]
            for od in range(10):
                dd=pbc.normalize()+timedelta(days=od)
                if dd>bc.normalize()+timedelta(days=1): break
                tt=load_ticks(dd)
                if len(tt)>0:
                    m=(tt["timestamp"]>pbc)&(tt["timestamp"]<=bc)
                    if m.sum()>0: tf.append(tt[m])
            if not tf: continue
            tdf=pd.concat(tf,ignore_index=True)
            if len(tdf)<2: continue
            lsv=None
            for i in range(len(tdf)):
                b,a,bv,av=tdf.iloc[i]["bid"],tdf.iloc[i]["ask"],tdf.iloc[i]["bid_vol"],tdf.iloc[i]["ask_vol"]
                ts=tdf.iloc[i]["timestamp"].value/1e6
                if pb is not None:
                    o=ofi(b,pb,a,pa,bv,pbv,av,pav); d=bv+av; su=o/(d+1e-8)
                    v=1.0/((ts-pts)+1e-3); spr=a-b
                    vals=[o,d,su,v,spr]
                    lsv=[zs[j].update(vals[j]) for j in range(5)]
                pb,pa,pbv,pav,pts=b,a,bv,av,ts
            if lsv:
                svecs.append(lsv); ovecs.append(ov); nc+=1
        if nc==0:
            results[f"w_{ws}"]={"error":"no_data"}; continue
        sa,oa=np.array(svecs),np.array(ovecs)
        ad=np.abs(sa-oa)
        wr={"n":nc,"features":{}}
        print(f"\n  {'Feat':>12} {'Mean|Δ|':>10} {'Max|Δ|':>10} {'KS':>8} {'p':>10}")
        for j,n in enumerate(feat_names):
            md,mx=float(ad[:,j].mean()),float(ad[:,j].max())
            ks,kp=ks_2samp(sa[:,j],oa[:,j])
            print(f"  {n:>12} {md:>10.4f} {mx:>10.4f} {ks:>8.4f} {kp:>10.6f}")
            wr["features"][n]={"mean_diff":round(md,6),"max_diff":round(mx,6),
                               "ks":round(float(ks),4),"ks_p":round(float(kp),6),"sig":bool(kp<0.05)}
        results[f"w_{ws}"]=wr
    with open(RESULTS_DIR/"streaming_vs_offline.json","w") as f: json.dump(results,f,indent=2)
    print(f"\n💾 Saved: streaming_vs_offline.json")

if __name__=="__main__": main()
