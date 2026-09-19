import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]
r=json.loads((HERE/'analysis/summary.json').read_text())
x=np.arange(6);cells=r['cells'];rows=r['rows']
fig,axs=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
for method,label,color in [('sequential','Sequential','#748094'),('pipeline','Pipeline','#3478b8'),
    ('draft_serial','Draft, serial','#d99a28'),('speculative','Speculative','#ca4c44')]:
    axs[0].plot(x,[row[method+'_ms'] for row in rows],marker='o',label=label,color=color)
axs[0].set_yscale('log');axs[0].set_ylabel('Complete query time (ms, log scale)')
axs[0].set_title('Four scheduling arms');axs[0].legend(fontsize=8)
for offset,base,label,color in [(-.19,'pipeline','vs pipeline','#3478b8'),
    (.19,r['strongest_fixed_control'],'vs '+r['strongest_fixed_control'],'#ca4c44')]:
    center=np.array([v['spec_speedup_vs_'+base] for v in rows])
    lo=np.array([v['spec_speedup_vs_'+base+'_min'] for v in rows])
    hi=np.array([v['spec_speedup_vs_'+base+'_max'] for v in rows])
    axs[1].bar(x+offset,center,.36,label=label,color=color,alpha=.85)
    axs[1].vlines(x+offset,lo,hi,color='black',linewidth=1)
    axs[1].scatter(x+offset,lo,color='black',marker='_',s=25)
    axs[1].scatter(x+offset,hi,color='black',marker='_',s=25)
axs[1].axhline(1,color='black',ls='--',lw=1)
axs[1].set_ylabel('Baseline time / speculative time')
axs[1].set_title('Speculative speedup; >1 is better')
axs[1].legend(fontsize=8)
cov={v['cell']:v for v in r['coverage']}
rate=np.array([cov[c]['drafted_verified_tiles']/cov[c]['verified_tiles'] for c in cells])*100
axs[2].bar(x,rate,color='#5b9c81',label='Drafted')
axs[2].bar(x,100-rate,bottom=rate,color='#d9e3eb',label='Repair')
axs[2].set_ylim(0,105);axs[2].set_ylabel('% of reliably undecided tiles')
axs[2].set_title('Coverage of the fixed sampled draft');axs[2].legend(fontsize=8)
for i,v in enumerate(rate):axs[2].text(i,v/2,f'{v:.1f}%',ha='center',va='center',fontsize=8)
for ax in axs:
    ax.set_xticks(x,cells,rotation=30);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('TensorJoin speculative execution probe | same GPU2 | complete sorted CPU output',fontsize=12)
fig.savefig(HERE/'analysis/speculative_probe.png',dpi=180)
fig.savefig(HERE/'analysis/speculative_probe.pdf')
