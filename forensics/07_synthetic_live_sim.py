"""
Forensic #7: Synthetic Live Market Simulation

Replays actual tick data through the EXACT live pipeline:
  1. Stream ticks one-by-one (simulating MT5 feed)
  2. Build Renko bricks from bid prices
  3. Compute features via streaming RollingZScore
  4. Maintain micro-buffer (deque maxlen=100)
  5. On brick close: snapshot → model inference → trade decision
  6. Execute trade with realistic latency/spread

Tests TWO configurations:
  A) Research config (W=1000, warmup=30) — should match backtest
  B) Trader config (W=5000, warmup=10000) — what actually runs live

Also injects synthetic latency (50ms, 100ms, 200ms) to measure decay.
"""
import sys,json,time,numpy as np,pandas as pd,tensorflow as tf
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from math import sqrt,log
from datetime import timedelta

BASE_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE_DIR/"src"))
TICK_DIR=BASE_DIR/"Data"/"Raw"/"Ticks"
OUTPUT_DIR=BASE_DIR/"outputs"
RESULTS_DIR=Path(__file__).resolve().parent/"results"
RESULTS_DIR.mkdir(parents=True,exist_ok=True)

class RZS:
    def __init__(self,w=1000):
        self.w=w;self.d=deque(maxlen=w);self.mean=0.0;self.M2=0.0
    def update(self,x):
        N=len(self.d)
        if N==self.w:
            xo=self.d[0];self.d.append(x);mn=self.mean+(x-xo)/N
            self.M2+=(x-xo)*((x-mn)+(xo-self.mean));self.mean=mn
            if self.M2<0:self.M2=0.0
            s=sqrt(self.M2/(N-1)) if N>1 else 0.0
            return 0.0 if s<1e-12 else (x-self.mean)/s
        else:
            self.d.append(x);N=len(self.d)
            if N<30:return 0.0
            a=list(self.d);self.mean=sum(a)/N;self.M2=sum((v-self.mean)**2 for v in a)
            s=sqrt(self.M2/(N-1)) if N>1 else 0.0
            return 0.0 if s<1e-12 else (x-self.mean)/s

def ofi(bk,bkm,ak,akm,bvk,bvkm,avk,avkm):
    db=bk-bkm;da=ak-akm
    return((1 if db>=0 else 0)*bvk-(1 if db<=0 else 0)*bvkm-(1 if da<=0 else 0)*avk+(1 if da>=0 else 0)*avkm)

class LiveRenkoBuilder:
    def __init__(self,brick_size_factor=0.0018):
        self.factor=brick_size_factor;self.current=None;self.uptrend=True;self.brick_size=None
    def update(self,bid):
        if self.current is None:
            self.current=bid;self.brick_size=bid*self.factor;return[]
        if self.brick_size is None:self.brick_size=bid*self.factor
        bricks=[]
        while True:
            d=bid-self.current
            if self.uptrend:
                if d>=self.brick_size:
                    self.current+=self.brick_size
                    bricks.append({"close":self.current,"dir":1,"size":self.brick_size})
                    self.brick_size=self.current*self.factor
                elif d<=-(self.brick_size*2):
                    self.uptrend=False;self.current-=(self.brick_size*2)
                    bricks.append({"close":self.current,"dir":-1,"size":self.brick_size})
                    self.brick_size=self.current*self.factor
                else:break
            else:
                if d<=-self.brick_size:
                    self.current-=self.brick_size
                    bricks.append({"close":self.current,"dir":-1,"size":self.brick_size})
                    self.brick_size=self.current*self.factor
                elif d>=(self.brick_size*2):
                    self.uptrend=True;self.current+=(self.brick_size*2)
                    bricks.append({"close":self.current,"dir":1,"size":self.brick_size})
                    self.brick_size=self.current*self.factor
                else:break
        return bricks

class LiveSimulator:
    def __init__(self,model,config,z_window=1000,warmup=30):
        self.model=model;self.cfg=config;self.z_window=z_window;self.warmup=warmup
        self.zs=[RZS(z_window) for _ in range(5)]
        self.renko=LiveRenkoBuilder()
        self.micro_buf=deque(maxlen=100)
        self.snapshots=deque(maxlen=10)
        self.macros=deque(maxlen=10)
        self.brick_sizes=deque(maxlen=50)
        self.brick_id=0;self.last_brick_ts=None
        self.brick_open=None;self.brick_size=None
        self.prev_brick_open=None;self.prev_brick_size=None
        self.pb=None;self.pa=None;self.pbv=None;self.pav=None;self.pts=None
        self.ticks_processed=0

    def process_tick(self,bid,ask,bid_vol,ask_vol,ts_ms):
        vec=None
        if self.pb is not None:
            o=ofi(bid,self.pb,ask,self.pa,bid_vol,self.pbv,ask_vol,self.pav)
            d=bid_vol+ask_vol;su=o/(d+1e-8)
            v=1.0/((ts_ms-self.pts)+1e-3);sp=ask-bid
            zvals=[self.zs[j].update(val) for j,val in enumerate([o,d,su,v,sp])]
            mid=(bid+ask)/2
            prog=(mid-self.brick_open)/self.brick_size if self.brick_open and self.brick_size else 0
            fz=0.0
            if self.prev_brick_open and self.prev_brick_size:
                if abs(mid-self.prev_brick_open)>=self.prev_brick_size:fz=1.0
            vec=zvals+[prog,1.0,fz,0.0]
            self.micro_buf.append((vec,self.brick_id))
        self.pb,self.pa,self.pbv,self.pav,self.pts=bid,ask,bid_vol,ask_vol,ts_ms
        self.ticks_processed+=1
        # Build renko
        bricks=self.renko.update(bid)
        signals=[]
        for brick in bricks:
            self.prev_brick_open=self.brick_open;self.prev_brick_size=self.brick_size
            self.brick_open=brick["close"]-(brick["dir"]*brick["size"])
            self.brick_size=brick["size"]
            # Snapshot
            snap=np.zeros((100,9),dtype=np.float32)
            bl=len(self.micro_buf)
            for j,(v,bi) in enumerate(self.micro_buf):
                r=100-bl+j;snap[r]=v
                snap[r,6]=1.0 if bi==self.brick_id else 0.0
                snap[r,8]=min((self.brick_id-bi)/100.0,1.0)
            self.snapshots.append(snap)
            # Macro
            dur=0.0
            if self.last_brick_ts:dur=(ts_ms-self.last_brick_ts)/1000.0
            self.last_brick_ts=ts_ms
            self.brick_sizes.append(brick["size"])
            z_sz=0.0
            if len(self.brick_sizes)>1:
                mu=np.mean(list(self.brick_sizes));sd=np.std(list(self.brick_sizes),ddof=1)
                if sd>1e-12:z_sz=(brick["size"]-mu)/sd
            self.macros.append([log(dur+1),float(brick["dir"]),z_sz])
            self.brick_id+=1
            # Inference
            if len(self.snapshots)==10 and self.ticks_processed>=self.warmup:
                mi=np.array(list(self.snapshots))[np.newaxis,...].astype(np.float32)
                ma=np.array(list(self.macros))[np.newaxis,...].astype(np.float32)
                preds=self.model([mi,ma],training=False)
                pwv=float(preds[0].numpy().flatten()[0])
                pov=float(preds[1].numpy().flatten()[0])
                if pwv>=self.cfg["Prob_Win_threshold"] and pov>=self.cfg["Pred_OS_threshold"]:
                    signals.append({"close":brick["close"],"dir":brick["dir"],
                        "size":brick["size"],"pw":pwv,"po":pov,"bid":bid,"ask":ask})
        return signals

def load_ticks(date):
    p=TICK_DIR/str(date.year)/f"{date.month:02d}"/f"{date.day:02d}.parquet"
    if not p.exists():return None
    df=pd.read_parquet(p)
    if df["timestamp"].dt.tz is None:df["timestamp"]=df["timestamp"].dt.tz_localize("UTC")
    return df

def main():
    print("="*60+"\n FORENSIC #7: Synthetic Live Market Simulation\n"+"="*60)
    model=tf.keras.models.load_model(OUTPUT_DIR/"model.keras")
    with open(OUTPUT_DIR/"config.json") as f:cfg=json.load(f)
    # Test both window configs
    configs=[
        {"name":"Research (W=1000,warm=30)","w":1000,"warm":30},
        {"name":"Trader (W=5000,warm=10000)","w":5000,"warm":10000},
    ]
    latencies_ms=[0,50,100,200]
    all_results={}
    # Load 2024 Q1 tick data (3 months)
    print("Loading 2024 Q1 tick data...")
    all_ticks=[]
    for month in range(1,4):
        for day in range(1,32):
            try:
                d=pd.Timestamp(2024,month,day,tz="UTC")
            except:continue
            t=load_ticks(d)
            if t is not None and len(t)>0:all_ticks.append(t)
    if not all_ticks:
        print("No tick data found!");return
    tick_df=pd.concat(all_ticks,ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(tick_df):,} ticks ({tick_df['timestamp'].min()} to {tick_df['timestamp'].max()})")
    bids=tick_df["bid"].values;asks=tick_df["ask"].values
    bvols=tick_df["bid_vol"].values;avols=tick_df["ask_vol"].values
    ts_ms_arr=tick_df["timestamp"].values.view("int64")/1e6
    for conf in configs:
        print(f"\n{'='*50}\n  Config: {conf['name']}\n{'='*50}")
        for lat in latencies_ms:
            sim=LiveSimulator(model,cfg,z_window=conf["w"],warmup=conf["warm"])
            trades=[];n_signals=0
            for i in range(len(bids)):
                sigs=sim.process_tick(bids[i],asks[i],bvols[i],avols[i],ts_ms_arr[i])
                for sig in sigs:
                    n_signals+=1
                    # Find entry tick (after latency)
                    entry_ts=ts_ms_arr[i]+lat
                    entry_idx=i
                    for ei in range(i+1,min(i+100,len(bids))):
                        if ts_ms_arr[ei]>=entry_ts:entry_idx=ei;break
                    if entry_idx>=len(bids)-10:continue
                    # Entry price (execution)
                    if sig["dir"]==1:ep=asks[entry_idx]
                    else:ep=bids[entry_idx]
                    bs=sig["size"];tp=ep+bs if sig["dir"]==1 else ep-bs
                    sl=ep-bs if sig["dir"]==1 else ep+bs
                    # Resolve
                    outcome=-1
                    for ri in range(entry_idx+1,min(entry_idx+5000,len(bids))):
                        sp=bids[ri] if sig["dir"]==1 else asks[ri]
                        if sig["dir"]==1:
                            if sp>=tp:outcome=1;break
                            if sp<=sl:outcome=0;break
                        else:
                            if sp<=tp:outcome=1;break
                            if sp>=sl:outcome=0;break
                    if outcome>=0:
                        trades.append({"outcome":outcome,"slippage":abs(ep-sig["close"]),
                            "spread":asks[entry_idx]-bids[entry_idx],"pw":sig["pw"],"po":sig["po"]})
                if i%500000==0 and i>0:print(f"    {i:,}/{len(bids):,} ticks...")
            key=f"{conf['name']}|lat={lat}ms"
            n_t=len(trades)
            if n_t>0:
                wr=np.mean([t["outcome"] for t in trades])
                avg_sl=np.mean([t["slippage"] for t in trades])
                avg_sp=np.mean([t["spread"] for t in trades])
            else:wr=0;avg_sl=0;avg_sp=0
            print(f"  Latency={lat}ms: {n_signals} signals, {n_t} resolved, WR={wr:.4f}, "
                  f"slip={avg_sl:.4f}, spread={avg_sp:.4f}")
            all_results[key]={"signals":n_signals,"trades":n_t,"wr":round(wr,6),
                "avg_slippage":round(avg_sl,6),"avg_spread":round(avg_sp,6),
                "config":conf["name"],"latency_ms":lat}
    with open(RESULTS_DIR/"synthetic_live_sim.json","w") as f:json.dump(all_results,f,indent=2)
    # Plot
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for conf in configs:
        xs,ys=[],[]
        for lat in latencies_ms:
            k=f"{conf['name']}|lat={lat}ms"
            if k in all_results and all_results[k]["trades"]>0:
                xs.append(lat);ys.append(all_results[k]["wr"])
        if xs:axes[0].plot(xs,ys,'o-',label=conf["name"],linewidth=2,markersize=8)
    axes[0].axhline(y=0.5,color='red',linestyle='--',alpha=0.5)
    axes[0].set_xlabel('Latency (ms)');axes[0].set_ylabel('Win Rate')
    axes[0].set_title('Live Sim: WR vs Latency by Config',fontweight='bold')
    axes[0].legend();axes[0].grid(True,alpha=0.3)
    # Trade counts
    for conf in configs:
        xs,ys=[],[]
        for lat in latencies_ms:
            k=f"{conf['name']}|lat={lat}ms"
            if k in all_results:xs.append(lat);ys.append(all_results[k]["trades"])
        if xs:axes[1].plot(xs,ys,'s-',label=conf["name"],linewidth=2,markersize=8)
    axes[1].set_xlabel('Latency (ms)');axes[1].set_ylabel('Trades Resolved')
    axes[1].set_title('Trade Count by Config',fontweight='bold')
    axes[1].legend();axes[1].grid(True,alpha=0.3)
    plt.tight_layout();plt.savefig(RESULTS_DIR/"synthetic_live_sim.png",dpi=150)
    print(f"\n💾 Saved: synthetic_live_sim.json, synthetic_live_sim.png")

if __name__=="__main__": main()
