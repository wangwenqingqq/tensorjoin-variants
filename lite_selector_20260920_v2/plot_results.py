"""Render the measured fixed-format crossover and the derived free-oracle gate.

Requires matplotlib 3.9.4. No GPU data, model training, or new timing is performed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize import gm, grouped_ratio

ROOT = Path(__file__).resolve().parent


def main():
    summary = json.loads((ROOT/'results/SUMMARY.json').read_text())
    out = ROOT/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':10,
                         'axes.labelsize':9, 'xtick.labelsize':8, 'ytick.labelsize':8,
                         'legend.fontsize':8, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.dpi':180})
    def save(fig, name):
        for ext in ('svg','png'):
            fig.savefig(out/(name+'.'+ext), bbox_inches='tight', metadata={'Creator':'TensorJoin experiment'})
        svg = out/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(7,5.5),sharey=True,layout='constrained')
    for ax, family in zip(axes.flat, ('clustered','outlier','logrange','mixed')):
        rows = [r for r in summary['configs'] if r['family'] == family]
        for fmt, marker, color, offset in [('int8','o','#0077BB',-.09),('fp16','s','#EE7733',.09)]:
            ratios = np.array([[r['process_medians_ms'][fmt][p]/r['process_medians_ms']['e3m4'][p]
                                for p in range(4)] for r in rows])
            center = np.array([gm(x) for x in ratios])
            errors = np.array([center-ratios.min(axis=1),ratios.max(axis=1)-center])
            ax.errorbar(np.arange(16)+offset,center,yerr=errors,fmt=marker,color=color,
                        ms=3,lw=.8,capsize=1.5,label=fmt.upper()+' / E3M4')
        ax.axhline(1,color='black',lw=.7)
        ax.axhline(1.05,color='0.55',ls=':',lw=.7)
        ax.axhline(1/1.05,color='0.55',ls=':',lw=.7)
        ax.set_title(family)
        ax.set_xticks([1.5,5.5,9.5,13.5],['101','102','103','201 (val)'])
        ax.set_xlabel('Parent seed (4 configurations each)')
        ax.set_xlim(-.6,15.6)
        for x in (3.5,7.5,11.5): ax.axvline(x,color='0.85',lw=.5)
    for ax in axes[:,0]: ax.set_ylabel('Host-to-host latency ratio')
    axes[0,0].legend(loc='upper left',frameon=False)
    save(fig,'format_crossover')
    # Fixed paths have process-level intervals. O2 is the frozen label-min oracle,
    # not an executed path; its scalar gate has no invented measurement interval.
    fig, ax = plt.subplots(figsize=(7,3),layout='constrained')
    names = ['INT8','E3M4','FP16','Free O2\n(label min)']
    colors = ['#0077BB','#EE7733','#777777','#FFFFFF']
    for split, offset, hatch in [('train',-.18,'///'),('validation',.18,'')]:
        part = [r for r in summary['configs'] if r['split'] == split]
        s = summary['splits'][split]; base = s['fixed_geomean_ms']['fp16']
        values = [s['fixed_geomean_ms'][m]/base for m in ('int8','e3m4','fp16')] + [s['free_O2_geomean_ms']/base]
        positions = np.arange(4)+offset
        ax.bar(positions,values,width=.34,color=colors,edgecolor='black',lw=.6,hatch=hatch,label=split)
        for i,m in enumerate(('int8','e3m4','fp16')):
            stat = grouped_ratio(part,m,'fp16');lo,hi=stat['log_t95_interval']
            ax.errorbar(positions[i],values[i],yerr=[[values[i]-lo],[hi-values[i]]],color='black',capsize=2,lw=.8)
        for x,v in zip(positions,values): ax.text(x,v+.032,f'{v:.3f}',ha='center',fontsize=7)
    ax.axhline(1,color='0.45',lw=.8)
    ax.axhline(1/1.10,color='black',ls='--',lw=.9)
    ax.text(3.55,1/1.10,'G1 target',ha='right',va='top',fontsize=8)
    ax.set_ylim(0,1.22);ax.set_xticks(np.arange(4),names)
    ax.set_ylabel('Geometric latency / fixed FP16')
    ax.legend(loc='upper left',ncols=2,frameon=False)
    save(fig,'oracle_headroom')
    print(json.dumps({'figures':2,'source':'results/SUMMARY.json','matplotlib':matplotlib.__version__}))


if __name__ == '__main__': main()
