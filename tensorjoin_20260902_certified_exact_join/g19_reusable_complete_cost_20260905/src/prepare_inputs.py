"""Reference the frozen G17 inputs and exact admitted G18 canonical IDs."""
from common import *
G18=P/'g18_rthiss_conservative_repair_20260905'
manifest=json.loads((P/'g17_rthiss_pair_contract_20260905/data/manifest_r1.json').read_text())
records=json.loads((G18/'results/closure_checks.json').read_text())['full_operator_observations']
for case in manifest['inputs']:
    prior=next(r for r in records if r['case']==case['name'])
    path=G18/'raw'/prior['record_id']/'pairs.u64'
    assert sha(path)==prior['sha256']
    case.update(g19_oracle_path=str(path.relative_to(P)),g19_oracle_sha256=sha(path),g19_oracle_count=prior['pairs'])
write(H/'artifacts/inputs.json',manifest)

