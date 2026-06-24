import os,json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR='/vertical_hopper_v3'; OUT_DIR='trajectories_nn'
FREQ_MAP={'050':1.00,'100':1.323,'125':1.39375,'150':1.63625,'175':1.90,'200':1.96333}
K=8;HIDDEN=256;LAYERS=4;BATCH=512;EPOCHS=300;LR=1e-3;W_PERIOD=0.5;W_DERIV=0.3
DEVICE=torch.device('cpu'); SEED=42
torch.manual_seed(SEED); np.random.seed(SEED); os.makedirs(OUT_DIR,exist_ok=True)

dfs=[]
for tag,hz in FREQ_MAP.items():
    df=pd.read_csv(f'CPG_orbit_bspline_cmaes_v3_{tag}.csv'); df['freq_hz']=hz; dfs.append(df)
data=pd.concat(dfs,ignore_index=True)
phi=data['Phase'].values.astype(np.float32); f_hz=data['freq_hz'].values.astype(np.float32)
hip=data['Hip'].values.astype(np.float32); knee=data['Knee'].values.astype(np.float32)
dhip=data['dHip_dphi'].values.astype(np.float32); dknee=data['dKnee_dphi'].values.astype(np.float32)
f_min,f_max=f_hz.min(),f_hz.max(); f_norm=(f_hz-f_min)/(f_max-f_min)

def fourier_embed(phi_arr,K):
    return np.stack([f(k*phi_arr) for k in range(1,K+1) for f in [np.cos,np.sin]],axis=1)

X=np.concatenate([fourier_embed(phi,K),f_norm[:,None]],axis=1).astype(np.float32)
Y=np.stack([hip,knee,dhip,dknee],axis=1)
Y_mean=Y.mean(0,keepdims=True); Y_std=Y.std(0,keepdims=True)+1e-8
Y_norm=((Y-Y_mean)/Y_std).astype(np.float32)
rng=np.random.RandomState(SEED)
idx=np.arange(len(X)); idx=rng.permutation(idx)
split=int(len(idx)*0.15)
idx_val=idx[:split]; idx_tr=idx[split:]
tt=lambda a: torch.tensor(a,dtype=torch.float32)
X_tr,Y_tr=tt(X[idx_tr]),tt(Y_norm[idx_tr]); X_val,Y_val=tt(X[idx_val]),tt(Y_norm[idx_val])
loader_tr=DataLoader(TensorDataset(X_tr,Y_tr),batch_size=BATCH,shuffle=True)
loader_val=DataLoader(TensorDataset(X_val,Y_val),batch_size=BATCH)

class ResBlock(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d)); self.act=nn.SiLU()
    def forward(self,x): return self.act(x+self.net(x))
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.inp=nn.Sequential(nn.Linear(2*K+1,HIDDEN),nn.SiLU())
        self.blocks=nn.ModuleList([ResBlock(HIDDEN) for _ in range(LAYERS)]); self.head=nn.Linear(HIDDEN,4)
    def forward(self,x):
        h=self.inp(x)
        for b in self.blocks: h=b(h)
        return self.head(h)
model=Net(); print(f'Params:{sum(p.numel() for p in model.parameters()):,}')

def ploss(m,fv):
    fs=tt(fv); p0=tt(fourier_embed(np.zeros(len(fv),np.float32),K)); p2=tt(fourier_embed(np.full(len(fv),2*np.pi,np.float32),K))
    return ((m(torch.cat([p0,fs[:,None]],1))[:,:2]-m(torch.cat([p2,fs[:,None]],1))[:,:2])**2).mean()

opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
fs_p=np.linspace(0,1,16,dtype=np.float32); hist={'tr':[],'val':[],'per':[]}

for ep in range(1,EPOCHS+1):
    model.train(); tl=0.0
    for xb,yb in loader_tr:
        p=model(xb); loss=((p[:,:2]-yb[:,:2])**2).mean()+W_DERIV*((p[:,2:]-yb[:,2:])**2).mean()+W_PERIOD*ploss(model,fs_p)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item()
    sched.step()
    model.eval(); vl=0.0
    with torch.no_grad():
        for xb,yb in loader_val: vl+=((model(xb)[:,:2]-yb[:,:2])**2).mean().item()
        pl=ploss(model,fs_p).item()
    hist['tr'].append(tl/len(loader_tr)); hist['val'].append(vl/len(loader_val)); hist['per'].append(pl)
    if ep%100==0: print(f'Ep{ep} tr={tl/len(loader_tr):.6f} val={vl/len(loader_val):.6f} per={pl:.8f}')

# Save model
meta={'K_fourier':K,'f_min':float(f_min),'f_max':float(f_max),'Y_mean':Y_mean.tolist(),'Y_std':Y_std.tolist(),'hidden':HIDDEN,'layers':LAYERS}
torch.save({'model_state':model.state_dict(),'meta':meta},f'{OUT_DIR}/ref_traj_nn.pt')
with open(f'{OUT_DIR}/ref_traj_nn_meta.json','w') as f: json.dump(meta,f,indent=2)

# Metrics
all_p,all_g=[],[]
with torch.no_grad():
    for xb,yb in loader_val:
        p=(model(xb).numpy()*Y_std+Y_mean); g=(yb.numpy()*Y_std+Y_mean)
        all_p.append(p[:,:2]); all_g.append(g[:,:2])
all_p=np.concatenate(all_p); all_g=np.concatenate(all_g)
metrics={}
for i,n in enumerate(['Hip','Knee']):
    mae=float(np.abs(all_p[:,i]-all_g[:,i]).mean()); rmse=float(np.sqrt(((all_p[:,i]-all_g[:,i])**2).mean()))
    metrics[n]={'MAE_rad':mae,'RMSE_rad':rmse}; print(f'{n}: MAE={mae:.5f} RMSE={rmse:.5f}')
per_err=float(ploss(model,np.linspace(0,1,50,dtype=np.float32)).item())
print(f'Periodicity err:{per_err:.8f}')

# ── Plot 1: Loss curves ──────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].semilogy(hist['tr'],label='Train (total)',alpha=0.8)
axes[0].semilogy(hist['val'],label='Val (position)',alpha=0.8)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (log)'); axes[0].set_title('Learning Curves'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
axes[1].semilogy(hist['per'],color='green',label='Periodicity loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss (log)'); axes[1].set_title('Periodicity Constraint'); axes[1].legend(); axes[1].grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/loss_curves.png',dpi=150); plt.close()

# ── Plot 2: Pred vs GT for each trained frequency ────────────────────────────
phi_d=np.linspace(0,2*np.pi,500,dtype=np.float32)
phi_fd=fourier_embed(phi_d,K)
fig,axes=plt.subplots(2,6,figsize=(24,7))
colors=['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4']
with torch.no_grad():
    for ci,(tag,hz) in enumerate(FREQ_MAP.items()):
        fn=float((hz-f_min)/(f_max-f_min))
        Xd=tt(np.concatenate([phi_fd,np.full((len(phi_d),1),fn,dtype=np.float32)],axis=1))
        pred=(model(Xd).numpy()*Y_std+Y_mean)
        df_gt=pd.read_csv(f'CPG_orbit_bspline_cmaes_v3_{tag}.csv')
        for ri,(col,gc) in enumerate([('Hip [rad]','Hip'),('Knee [rad]','Knee')]):
            ax=axes[ri][ci]
            ax.plot(df_gt['Phase'],df_gt[gc],'k-',lw=2,label='GT (CMA-ES)')
            ax.plot(phi_d,pred[:,ri],'--',color=colors[ci],lw=2,label='NN pred')
            ax.set_title(f'{hz:.2f} Hz',fontsize=10); ax.set_xlabel('Phase φ'); ax.set_ylabel(col,fontsize=8)
            ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
axes[0][0].set_ylabel('Hip angle [rad]',fontsize=9); axes[1][0].set_ylabel('Knee angle [rad]',fontsize=9)
plt.suptitle('Reference Trajectory NN — Trained Frequencies (GT vs Prediction)',fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/traj_comparison.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Plot 3: Interpolation at unseen frequencies ──────────────────────────────
unseen=[1.125,1.375,1.625,1.875]
fig,axes=plt.subplots(2,4,figsize=(18,7))
with torch.no_grad():
    for ci,hz in enumerate(unseen):
        fn=float((hz-f_min)/(f_max-f_min))
        Xd=tt(np.concatenate([phi_fd,np.full((len(phi_d),1),fn,dtype=np.float32)],axis=1))
        pred=(model(Xd).numpy()*Y_std+Y_mean)
        hz_lo=max([h for h in FREQ_MAP.values() if h<=hz],default=None)
        hz_hi=min([h for h in FREQ_MAP.values() if h>=hz],default=None)
        for ri,col in enumerate(['Hip [rad]','Knee [rad]']):
            ax=axes[ri][ci]
            ax.plot(phi_d,pred[:,ri],'b-',lw=2.5,label=f'NN {hz}Hz (unseen)')
            for hzr in [hz_lo,hz_hi]:
                if hzr is None: continue
                tagr=[t for t,h in FREQ_MAP.items() if h==hzr][0]
                df_gt=pd.read_csv(f'CPG_orbit_bspline_cmaes_v3_{tagr}.csv')
                gc='Hip' if ri==0 else 'Knee'
                ax.plot(df_gt['Phase'],df_gt[gc],'--',lw=1.2,color='gray',alpha=0.7,label=f'GT {hzr:.2f}Hz')
            ax.set_title(f'{hz} Hz (interpolated)',fontsize=9); ax.set_xlabel('Phase φ'); ax.set_ylabel(col,fontsize=8)
            ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
plt.suptitle('Interpolation to Unseen Frequencies (1.125 / 1.375 / 1.625 / 1.875 Hz)',fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/interpolation.png',dpi=150,bbox_inches='tight'); plt.close()

print('All outputs saved.')