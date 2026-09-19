import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'analysis/summary.json').read_text())
hist=json.loads((HERE/'analysis/histograms.json').read_text())
names=['original','random','kd_variance','pca_sort']
labels=['Original','Random','Geometry tree','PCA sort']
colors=['#677580','#c09847','#168c83','#546ac2']

def find(layout,kind,cell,size):
    return next(x for x in rows if (x['layout'],x['kind'],x['cell'],x['size'])==(layout,kind,cell,size))

fig,axs=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
ax=axs[0,0]
for cell,label,c in [('k4','k4','#168c83'),('original','Original threshold','#546ac2'),('k1024','k1024','#c09847')]:
    ys=[100*find('original',f'r{r}',cell,64)['candidate_pair_fraction'] for r in [16,32,64]]
    ax.plot([16,32,64],ys,'o-',color=c,label=label)
ax.set(yscale='log',xlabel='PCA dimensions',ylabel='Surviving pairs (%)',title='A  Pair filtering: full census')
ax.set_xticks([16,32,64]);ax.legend(frameon=False);ax.grid(alpha=.2)

ax=axs[0,1];x=np.arange(4);w=.2
for j,(kind,size,label,c) in enumerate([('r64',64,'Projected: 64 tiles','#546ac2'),('r64',16,'Projected: 16 tiles','#168c83'),('oracle',64,'True hits: 64 tiles','#b4bac2')]):
    ax.bar(x+(j-1)*w,[100*find(n,kind,'original',size)['occupied_capacity_fraction'] for n in names],width=w,label=label,color=c)
ax.axhline(100*find('original','r64','original',64)['candidate_pair_fraction'],color='#bc5659',ls='--',lw=1.5,label='Projected surviving pairs')
ax.set_xticks(x,labels);ax.set(ylabel='Retained pair capacity (%)',title='B  Execution granularity: original threshold',ylim=(0,105))
ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2)

ax=axs[1,0]
for n,label,c in zip(names,labels,colors):
    h=np.array(hist[f'{n}/r64/original/pairs64'])
    ax.plot(np.arange(1,4097),100*np.cumsum(h)[1:]/h.sum(),color=c,label=label)
ax.axvline(64,color='#999999',ls='--',lw=1)
ax.set(xscale='log',xlabel='Surviving pairs per occupied 64 tile',ylabel='Cumulative occupied tiles (%)',title='C  How sparse are the exceptions?',ylim=(0,101),xlim=(1,4096))
ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)

ax=axs[1,1]
for n,label,c in zip(names,labels,colors):
    h=np.array(hist[f'{n}/r64/original/children64_16'])
    cdf=100*np.cumsum(h)/h.sum()
    ax.step(np.r_[np.arange(17),16.5],np.r_[cdf,cdf[-1]],where='post',color=c,label=label)
ax.set(xlabel='Occupied 16 children per occupied 64 tile',ylabel='Cumulative occupied parents (%)',title='D  Are survivors concentrated in subtiles?',ylim=(0,101),xlim=(.5,16.5))
ax.set_xticks([1,4,8,12,16]);ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)
fig.suptitle('TensorJoin: projected survivors and multiscale structure\nB-D: rank 64, original threshold; C-D: full occupied off-diagonal tiles\nCounts only; no GPU speedup is measured',fontsize=12)
for ext in ['png','pdf','svg']:fig.savefig(HERE/'analysis'/f'multiscale_probe.{ext}',dpi=180)
