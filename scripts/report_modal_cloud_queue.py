"""Read cloud progress and render native scores without dispatching any games."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import modal_panel_cloud_queue as cloud
from modal_panel_cloud_app import deployment_environment, deployment_names, deployment_worker


def project(snapshot, receipts):
    """Preserve the cutover's accepted rows and add fully validated cloud rows."""
    rows={r['label']:copy.deepcopy(r['initial_row']) for r in snapshot['games']}
    definitions={r['label']:r for r in snapshot['games']}
    queue_sha=cloud.digest(cloud.encoded(snapshot))
    for label,receipt in receipts.items():
        if label not in rows:raise ValueError('cloud result absent from frozen panel')
        definition=definitions[label]
        if receipt.get('status')=='quarantined':
            cloud.validate_exclusion(definition,receipt,queue_sha)
            rows[label].update(status='quarantined',result=None,cloud_exclusion=receipt)
            continue
        if (receipt['schema']!=cloud.SCHEMA+'.acceptance' or receipt['status']!='complete'
                or receipt['queue_sha256']!=queue_sha or receipt['label']!=label
                or receipt['plan_sha256']!=cloud.receipt_plan_sha(definition,receipt)):
            raise ValueError('cloud acceptance binding differs')
        if definition['initial_status']!='pending':raise ValueError('protected result cannot be replaced')
        rows[label].update(status='complete',result=receipt['native_acceptance'],
            cloud_provenance={'queue_sha256':queue_sha,'plan_sha256':receipt['plan_sha256'],
                'call_id':receipt['manifest']['call_id'],'execution':receipt['manifest']['execution'],
                'physical_attempts':len(receipt['physical_history'])+sum(len(f['physical_history'])
                    for f in (receipt.get('retry_authorization') or {}).get('prior_failures',[]))})
    return rows


def markdown(snapshot,rows,health):
    games=snapshot['manifest']['games'];done=sum(r['status']=='complete' for r in rows.values())
    lines=['# Slippi-AI public checkpoint benchmark','',f'Accepted {done}/{len(games)}.',
        '', 'Stocks are taken / conceded from Frisson’s perspective. SLP-only; original paired ports and native policy settings.',
        '',f"Cloud dispatcher: {health.get('status','unobserved')}. Retained errors: {len(health.get('errors',[]))}.",'']
    exclusions=[label for label,row in rows.items() if row.get('cloud_exclusion')]
    if exclusions:
        lines += ['Unscored infrastructure exclusions (excluded from W-L and stock totals):',
            '',*[f'- `{label}`: three failed starts; all evidence retained.' for label in exclusions],'']
    if health.get('held_labels'):
        lines += ['Unresolved scheduling holds (unscored; remaining games continue):','',
            *[f'- `{label}`' for label in health['held_labels']],'']
    for profile in ('75m','10m'):
        lines += [f'## {profile.upper()} RL','','| Opponent | Block | Complete | W-L | Stocks taken / conceded |',
            '| --- | --- | ---: | ---: | ---: |']
        for release in snapshot['manifest']['releases']:
            for block in ('supported-mirror','extended-roster','forced-mirror'):
                specs=[g for g in games if (g['profile'],g['release'],g['block'])==(profile,release,block)]
                accepted=[rows[g['label']]['result'] for g in specs if rows[g['label']]['status']=='complete']
                wins=sum(r['win'] for r in accepted)
                lines.append(f"| {release} | {block} | {len(accepted)}/{len(specs)} | {wins}-{len(accepted)-wins} | {sum(r['stocks_taken'] for r in accepted)} / {sum(r['stocks_conceded'] for r in accepted)} |")
        lines.append('')
    return '\n'.join(lines)+'\n'


def report(directory):
    import modal
    directory=Path(directory).absolute();body=(directory/'metadata/snapshot.json').read_bytes()
    snapshot=cloud.validate_snapshot(json.loads(body));sha=cloud.digest(body)
    deployment_path=directory/'deployment.json'
    deployment=json.loads(deployment_path.read_bytes()) if deployment_path.exists() else {}
    if deployment and deployment.get('snapshot_sha256')!=sha:
        raise ValueError('report deployment belongs to a different snapshot')
    names=deployment_names(deployment)
    store=modal.Dict.from_name(names['state_name'],environment_name=deployment_environment(deployment))
    entries=dict(store.items());prefix=sha+'/result/'
    receipts={k[len(prefix):]:v for k,v in entries.items() if k.startswith(prefix)}
    rows=project(snapshot,receipts);health=entries.get(sha+'/health',{})
    output={'schema':cloud.SCHEMA+'.projection','updated_unix':time.time(),'queue_sha256':sha,
        **names,
        'durable_module':deployment_worker(deployment),
        'environment_name':deployment_environment(deployment),
        'health':health,'games':rows,'counts':{s:sum(r['status']==s for r in rows.values()) for s in ('complete','pending','quarantined')}}
    for filename,value in [('CLOUD_STATE.json',cloud.encoded(output)),('CLOUD_RESULTS.md',markdown(snapshot,rows,health).encode())]:
        temporary=directory/('.'+filename+'.tmp');temporary.write_bytes(value);temporary.replace(directory/filename)
    print(json.dumps({'counts':output['counts'],'health':health}),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    report(parser.parse_args().directory)
