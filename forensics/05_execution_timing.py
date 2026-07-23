"""
Forensic #5: Execution Timing Scenarios
Tests entry at different tick offsets after brick close with model filtering.
"""
import sys,json,numpy as np,pandas as pd,tensorflow as tf
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta

BASE_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE_DIR/"src"))
TICK_DIR=BASE_DIR/"Data"/"Raw"/"Ticks"
OUTPUT_DIR=BASE_DIR/"outputs"
TENSOR_DIR=OUTPUT_DIR/"tensors"
RESULTS_DIR=Path(__file__).resolve().parent/"results"
RESULTS_DIR.mkdir(parents=True,exist_ok=True)

def load_ticks(date):
    p=TICK_DIR/str(date.year)/f"{date.month:02d}"/f"{date.day:02d}.parquet"
    if not p.exists(): return pd.DataFrame(columns=["timestamp","bid","bid_vol","ask","ask_vol"])
    df=pd.read_parquet(p)
    if df["timestamp"].dt.tz is None: df["timestamp"]=df["timestamp"].dt.tz_localize("UTC")
    return df

def load_ticks_after(t,n=60):
    cd=t.normalize();frames=[]
    for o in range(10):
        dd=cd+timedelta(days=o);tk=load_ticks(dd)
        if len(tk)>0:
            f=tk[tk["timestamp"]>t]
            if len(f)>0: frames.append(f)
            if sum(len(x) for x in frames)>=n: break
    if not frames: return pd.DataFrame(columns=["timestamp","bid","bid_vol","ask","ask_vol"])
    return pd.concat(frames,ignore_index=True).sort_values("timestamp").head(n).reset_index(drop=True)

def resolve(entry,bs,long,ticks):
    tp=entry+bs if long else entry-bs
    sl=entry-bs if long else entry+bs
    for i in range(len(ticks)):
        p=ticks.iloc[i]["bid"] if long else ticks.iloc[i]["ask"]
        if long:
            if p>=tp: return 1
            if p<=sl: return 0
        else:
            if p<=tp: return 1
            if p>=sl: return 0
    return -1

def main():
    print("="*60+"\n FORENSIC #5: Execution Timing Scenarios\n"+"="*60)
    # Load model and predict on holdout
    model=tf.keras.models.load_model(OUTPUT_DIR/"model.keras")
    micro=np.load(TENSOR_DIR/"holdout_micro.npy")
    macro=np.load(TENSOR_DIR/"holdout_macro.npy")
    y_class=np.load(TENSOR_DIR/"holdout_y_class.npy")
    with open(OUTPUT_DIR/"config.json") as f: cfg=json.load(f)
    # Predict
    pw,po=[],[]
    for i in range(0,len(micro),64):
        e=min(i+64,len(micro))
        p=model([micro[i:e],macro[i:e]],training=False)
        pw.append(p[0].numpy().flatten());po.append(p[1].numpy().flatten())
    pw,po=np.concatenate(pw),np.concatenate(po)
    # Signal mask
    sig_mask=(pw>=cfg["Prob_Win_threshold"])&(po>=cfg["Pred_OS_threshold"])
    baseline_wr=np.mean(y_class[sig_mask])
    n_sig=np.sum(sig_mask)
    print(f"Baseline: {n_sig} trades, WR={baseline_wr:.4f}")
    # Load holdout labels for dates
    hlp=OUTPUT_DIR/"holdout"/"labels.parquet"
    if not hlp.exists(): hlp=OUTPUT_DIR/"labels.parquet"
    hl=pd.read_parquet(hlp);hl["date"]=pd.to_datetime(hl["date"],utc=True)
    if "exclude_flag" in hl.columns: hl=hl[~hl["exclude_flag"]]
    if hl["date"].min().year<2024: hl=hl[hl["date"]>="2024-01-01"]
    hl=hl[hl["y_class"].notna()].reset_index(drop=True)
    # Align indices - take first min(len,len) samples
    n=min(len(hl),len(sig_mask))
    offsets=[0,1,2,3,5,10]
    res={k:{"wins":0,"total":0,"slippage":[],"latency":[]} for k in offsets}
    # Also test spread-aware: only enter if spread < median
    res["spread_aware"]={"wins":0,"total":0,"slippage":[],"latency":[]}
    all_spreads=[]
    processed=0
    for idx in range(n):
        if not sig_mask[idx]: continue
        row=hl.iloc[idx] if idx<len(hl) else None
        if row is None: continue
        bc=row["date"];bs=float(row["brick_size"]);is_long=bool(row["uptrend"])
        theo_entry=float(row["close"])
        ft=load_ticks_after(bc,60)
        if len(ft)<25: continue
        spread0=float(ft.iloc[0]["ask"]-ft.iloc[0]["bid"])
        all_spreads.append(spread0)
        for off in offsets:
            if off>=len(ft)-5: continue
            tk=ft.iloc[off]
            ep=float(tk["ask"]) if is_long else float(tk["bid"])
            lat=(tk["timestamp"]-ft.iloc[0]["timestamp"]).total_seconds()*1000 if off>0 else 0
            scan=ft.iloc[off+1:]
            if len(scan)<3: continue
            o=resolve(ep,bs,is_long,scan)
            if o>=0:
                res[off]["total"]+=1;res[off]["wins"]+=o
                res[off]["slippage"].append(abs(ep-theo_entry))
                res[off]["latency"].append(lat)
        # Spread-aware: find first tick with spread < median (use running median)
        if len(all_spreads)>20:
            med_spread=np.median(all_spreads[-100:])
            for si in range(min(15,len(ft)-5)):
                s=float(ft.iloc[si]["ask"]-ft.iloc[si]["bid"])
                if s<=med_spread:
                    ep=float(ft.iloc[si]["ask"]) if is_long else float(ft.iloc[si]["bid"])
                    lat=(ft.iloc[si]["timestamp"]-ft.iloc[0]["timestamp"]).total_seconds()*1000
                    scan=ft.iloc[si+1:]
                    if len(scan)>=3:
                        o=resolve(ep,bs,is_long,scan)
                        if o>=0:
                            res["spread_aware"]["total"]+=1;res["spread_aware"]["wins"]+=o
                            res["spread_aware"]["slippage"].append(abs(ep-theo_entry))
                            res["spread_aware"]["latency"].append(lat)
                    break
        processed+=1
        if processed%200==0: print(f"  {processed} signals processed")
    print(f"\nProcessed {processed} signals")
    # Results
    print("\n"+"="*60+"\n EXECUTION TIMING RESULTS\n"+"="*60)
    out={}
    print(f"\n{'Scenario':<16}{'N':>8}{'WR':>10}{'Avg Slip':>12}{'Med Lat ms':>14}")
    print("-"*60)
    for k in list(offsets)+["spread_aware"]:
        d=res[k]
        if d["total"]==0: continue
        wr=d["wins"]/d["total"]
        sl=np.mean(d["slippage"]) if d["slippage"] else 0
        lt=np.median(d["latency"]) if d["latency"] else 0
        label=f"t+{k}" if isinstance(k,int) else k
        print(f"  {label:<14}{d['total']:>8}{wr:>10.4f}{sl:>12.4f}{lt:>14.1f}")
        out[str(k)]={"n":d["total"],"wr":round(wr,6),"avg_slip":round(sl,6),"med_lat":round(lt,2)}
    out["baseline_wr"]=round(float(baseline_wr),6)
    out["baseline_n"]=int(n_sig)
    with open(RESULTS_DIR/"execution_timing.json","w") as f: json.dump(out,f,indent=2)
    # Plot
    fig,ax=plt.subplots(figsize=(10,6))
    xs,ys=[],[]
    for k in offsets:
        if str(k) in out:
            xs.append(k);ys.append(out[str(k)]["wr"])
    if xs:
        ax.plot(xs,ys,'o-',color='steelblue',linewidth=2,markersize=8,label='Market Order')
        if "spread_aware" in out:
            ax.axhline(y=out["spread_aware"]["wr"],color='green',linestyle='--',label=f'Spread-Aware: {out["spread_aware"]["wr"]:.4f}')
        ax.axhline(y=0.5,color='red',linestyle='--',alpha=0.5,label='Break-even')
        ax.set_xlabel('Tick Offset');ax.set_ylabel('Win Rate')
        ax.set_title('Execution Timing: WR by Entry Delay',fontweight='bold')
        ax.legend();ax.grid(True,alpha=0.3)
    plt.tight_layout();plt.savefig(RESULTS_DIR/"execution_timing.png",dpi=150)
    print(f"\n💾 Saved: execution_timing.json, execution_timing.png")

if __name__=="__main__": main()
