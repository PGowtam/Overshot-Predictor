"""
Forensic #6: Edge Localization — Feature Attribution
Which features drive high Pred_OS predictions?
"""
import sys,json,numpy as np,tensorflow as tf
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp

BASE_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE_DIR/"src"))
OUTPUT_DIR=BASE_DIR/"outputs"
TENSOR_DIR=OUTPUT_DIR/"tensors"
RESULTS_DIR=Path(__file__).resolve().parent/"results"
RESULTS_DIR.mkdir(parents=True,exist_ok=True)

FEAT_NAMES=["z_OFI","z_Depth","z_Susc","z_Vel","z_Spread","Progress","Flag_Curr","Flag_Zone","Decay"]

def main():
    print("="*60+"\n FORENSIC #6: Edge Localization\n"+"="*60)
    model=tf.keras.models.load_model(OUTPUT_DIR/"model.keras")
    micro=np.load(TENSOR_DIR/"holdout_micro.npy")
    macro=np.load(TENSOR_DIR/"holdout_macro.npy")
    y_class=np.load(TENSOR_DIR/"holdout_y_class.npy")
    y_mag=np.load(TENSOR_DIR/"holdout_y_mag.npy")
    with open(OUTPUT_DIR/"config.json") as f: cfg=json.load(f)
    # Predict
    pw,po=[],[]
    for i in range(0,len(micro),64):
        e=min(i+64,len(micro))
        p=model([micro[i:e],macro[i:e]],training=False)
        pw.append(p[0].numpy().flatten());po.append(p[1].numpy().flatten())
    pw,po=np.concatenate(pw),np.concatenate(po)

    # Split into high-confidence vs rejected
    high_mask=(pw>=cfg["Prob_Win_threshold"])&(po>=1.6)
    low_mask=po<1.0
    print(f"High-confidence (PO>=1.6): {np.sum(high_mask)}")
    print(f"Rejected (PO<1.0): {np.sum(low_mask)}")

    # Extract last 10 ticks of latest snapshot for each
    # micro shape: (N,10,100,9) -> latest snapshot: [:,9,:,:] -> last 10 ticks: [:,9,90:,:]
    latest=micro[:,-1,-10:,:]  # (N,10,9)
    feat_means=np.mean(latest,axis=1)  # (N,9)

    results={"n_high":int(np.sum(high_mask)),"n_low":int(np.sum(low_mask)),"features":{}}

    print(f"\n{'Feature':>12} {'High Mean':>10} {'Low Mean':>10} {'Δ':>8} {'KS':>8} {'p':>10}")
    print("-"*65)
    for j,name in enumerate(FEAT_NAMES):
        h=feat_means[high_mask,j]; l=feat_means[low_mask,j]
        hm,lm=h.mean(),l.mean()
        ks,kp=ks_2samp(h,l) if len(h)>10 and len(l)>10 else (0,1)
        print(f"  {name:>12} {hm:>10.4f} {lm:>10.4f} {hm-lm:>+8.4f} {ks:>8.4f} {kp:>10.2e}")
        results["features"][name]={"high_mean":round(float(hm),4),"low_mean":round(float(lm),4),
            "delta":round(float(hm-lm),4),"ks":round(float(ks),4),"ks_p":round(float(kp),8)}

    # Gradient-based attribution for Pred_OS head
    print("\n📊 Gradient Attribution (Pred_OS head):")
    sample_idx=np.where(high_mask)[0][:200]
    if len(sample_idx)>0:
        mi_t=tf.constant(micro[sample_idx],dtype=tf.float32)
        ma_t=tf.constant(macro[sample_idx],dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(mi_t)
            preds=model([mi_t,ma_t],training=False)
            pred_os_out=preds[1]
        grads=tape.gradient(pred_os_out,mi_t).numpy()  # (N,10,100,9)
        # Mean absolute gradient per feature
        mean_grads=np.mean(np.abs(grads),axis=(0,1,2))  # (9,)
        grad_results={}
        sorted_idx=np.argsort(-mean_grads)
        print(f"  {'Feature':>12} {'Mean|∇|':>12} {'Rank':>6}")
        for rank,j in enumerate(sorted_idx):
            print(f"  {FEAT_NAMES[j]:>12} {mean_grads[j]:>12.6f} {rank+1:>6}")
            grad_results[FEAT_NAMES[j]]={"mean_abs_grad":round(float(mean_grads[j]),8),"rank":rank+1}
        results["gradient_attribution"]=grad_results

    # Occlusion importance for Pred_OS
    print("\n📊 Occlusion Importance (Pred_OS head):")
    base_po=po[high_mask].mean()
    occ_results={}
    for j,name in enumerate(FEAT_NAMES):
        m=micro[high_mask].copy()
        m[:,:,:,j]=0.0  # Zero out feature j
        po2=[]
        for i in range(0,len(m),64):
            e=min(i+64,len(m))
            p=model([m[i:e],macro[high_mask][i:e]],training=False)
            po2.append(p[1].numpy().flatten())
        po2=np.concatenate(po2)
        delta=base_po-po2.mean()
        print(f"  {name:>12}: ΔPred_OS = {delta:>+.4f}")
        occ_results[name]=round(float(delta),6)
    results["occlusion_importance"]=occ_results

    # WR analysis: high-conf WINs vs LOSSes feature comparison
    high_win=(high_mask)&(y_class==1)
    high_loss=(high_mask)&(y_class==0)
    if np.sum(high_win)>5 and np.sum(high_loss)>5:
        print(f"\n📊 Feature differences: High-conf WINS vs LOSSES")
        print(f"  Wins: {np.sum(high_win)}, Losses: {np.sum(high_loss)}")
        win_loss_delta={}
        for j,name in enumerate(FEAT_NAMES):
            wm=feat_means[high_win,j].mean()
            lm2=feat_means[high_loss,j].mean()
            print(f"  {name:>12}: Win={wm:.4f}, Loss={lm2:.4f}, Δ={wm-lm2:+.4f}")
            win_loss_delta[name]={"win_mean":round(float(wm),4),"loss_mean":round(float(lm2),4)}
        results["win_vs_loss_features"]=win_loss_delta

    with open(RESULTS_DIR/"edge_localization.json","w") as f: json.dump(results,f,indent=2)

    # Plot
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    # Feature importance by occlusion
    names=list(occ_results.keys());vals=[occ_results[n] for n in names]
    si=np.argsort(np.abs(vals))[::-1]
    axes[0].barh([names[i] for i in si],[vals[i] for i in si],color='steelblue')
    axes[0].set_xlabel('ΔPred_OS when zeroed');axes[0].set_title('Occlusion Importance',fontweight='bold')
    axes[0].grid(True,alpha=0.3,axis='x')
    # Gradient attribution
    if "gradient_attribution" in results:
        gn=list(results["gradient_attribution"].keys())
        gv=[results["gradient_attribution"][n]["mean_abs_grad"] for n in gn]
        gi=np.argsort(gv)[::-1]
        axes[1].barh([gn[i] for i in gi],[gv[i] for i in gi],color='darkorange')
        axes[1].set_xlabel('Mean |∇|');axes[1].set_title('Gradient Attribution',fontweight='bold')
        axes[1].grid(True,alpha=0.3,axis='x')
    # Feature distribution: high vs low
    for j in [0,2,4]:
        axes[2].hist(feat_means[high_mask,j],bins=30,alpha=0.4,label=f'{FEAT_NAMES[j]} (high)',density=True)
        axes[2].hist(feat_means[low_mask,j],bins=30,alpha=0.4,label=f'{FEAT_NAMES[j]} (low)',density=True,linestyle='--')
    axes[2].set_title('Feature Dist: High vs Low Confidence',fontweight='bold')
    axes[2].legend(fontsize=7);axes[2].grid(True,alpha=0.3)
    plt.tight_layout();plt.savefig(RESULTS_DIR/"edge_localization.png",dpi=150)
    print(f"\n💾 Saved: edge_localization.json, edge_localization.png")

if __name__=="__main__": main()
