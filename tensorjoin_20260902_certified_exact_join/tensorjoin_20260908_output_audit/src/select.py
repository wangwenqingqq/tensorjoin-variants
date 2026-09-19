from common import *
def main():
 costs={o:[] for o in OUTPUTS}
 for index in range(2):
  r=json.loads((HERE/'results'/f'explore_{index:02d}.json').read_text());assert r['pass']
  for c in CELLS:
   for m in METHODS:
    for o in OUTPUTS:
     values=[s['seconds'] for s in r['samples'] if s['phase']=='retained' and s['cell']==c['name'] and s['method']==m and s['output_mode']==o];assert len(values)==2
     costs[o].append(float(np.median(values)))
 scores={o:float(np.mean(v)) for o,v in costs.items()};chosen=min(OUTPUTS,key=lambda o:scores[o]);challenger=chosen if chosen!='legacy' else min(OUTPUTS[1:],key=lambda o:scores[o])
 rec=dict(created=time.time(),scores=scores,chosen=chosen,challenger=challenger,rule='Lowest arithmetic mean of process medians over 18 equally weighted method/cell combinations.',exploration_sha256={f'explore_{i:02d}.json':sha(HERE/'results'/f'explore_{i:02d}.json') for i in range(2)},timing_freeze_sha256=sha(HERE/'artifacts/timing_freeze.json'))
 write_json(HERE/'artifacts/selection.json',rec);print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
