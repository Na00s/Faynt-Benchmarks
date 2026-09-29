"""Prepare and run bounded original single-game claims with cloud persistence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import modal_panel_durable_game as d
import modal_panel_queue as q
import ingest_modal_durable_game as importer

SCHEMA = "e011.modal-durable-cohort.v1"


def claim_when_available(journal, journal_sha, pair_id, frozen, panel):
    """Retry only lock contention or the state changing before lock acquisition."""
    for attempt in range(40):
        _, state_sha = q.current_state(frozen, panel)
        try:
            return q.claim_pair(journal, journal_sha, pair_id,
                expected_state_sha256=state_sha, first_pending_only=True)
        except BlockingIOError:
            if attempt == 39: raise
        except ValueError as error:
            if str(error) != 'claim requires current exact state SHA' or attempt == 39: raise
        time.sleep(0.25)


def ingest_when_available(row, journal, journal_sha):
    """Exact-once import permits a bounded retry after local lock contention."""
    for attempt in range(40):
        try:
            return importer.ingest(row['directory'], journal, journal_sha,
                row['plan_sha256'], row['claim_token'])
        except BlockingIOError:
            if attempt == 39: raise
            time.sleep(0.25)


def eligible_pairs(frozen, state, journal):
    pairs = frozen['pairs']
    for index, pair in enumerate(pairs):
        pending = [g for g in pair['games'] if state['games'][g['label']]['status'] == 'pending']
        if not pending: continue
        blocked = any(any(state['games'][g['label']]['status'] not in {'complete', 'quarantined'} for g in earlier['games'])
            for earlier in pairs[:index] if earlier['phase'] < pair['phase'] or
            earlier['phase'] == pair['phase'] and earlier['worker_slot'] == pair['worker_slot'])
        if blocked: continue
        if any(not (journal/'ingests'/a['attempt_token']/'done.json').is_file()
               for claim in q.claims(journal, pair['pair_id']) for a in claim['attempts']): continue
        yield pair


def prepare(output, journal, journal_sha, *, limit=64):
    if type(limit) is not int or not 1 <= limit <= 64: raise ValueError('one to64 games required')
    output = Path(output).absolute(); journal = q.journal_path(journal)
    if output.exists(): raise ValueError('fresh durable cohort output required')
    frozen, panel = q.load_plan(journal, journal_sha); state, _ = q.current_state(frozen, panel)
    pairs = list(eligible_pairs(frozen, state, journal))[:limit]
    if not pairs: raise ValueError('no eligible free original worker slots')
    output.mkdir(parents=True)
    template = q.read(d.batch.base.COMPAT/'phase1-ready-v1/937db7c201b64357/authorization.json')
    tape = d.batch.base.COMPAT/'live-pilot-v2'
    lookup = {g['label']: g for g in panel['games']}
    rows=[]
    for pair in pairs:
        labels = [g['label'] for g in pair['games']]
        index = next(i for i,n in enumerate(labels) if state['games'][n]['status']=='pending')
        attempts=[]
        for label in labels:
            old=state['games'][label]
            number=len(old['attempts'])+1 if old['status']=='pending' else max(1,len(old['attempts']))
            attempts.append(f'{label}-a{number}')
        bound=d.bind(labels,attempts,index)
        directory=output/pair['pair_id']/'run';directory.mkdir(parents=True)
        authorization={**template,'scope':bound.fixed_scope(),'limits':bound.limits(),
                       'bindings':{**template['bindings'],'checkpoints':bound.CHECKPOINTS}}
        auth=directory.parent/'authorization.json';q.new_file(auth,authorization)
        plan=bound.make_plan(directory,tape,auth);q.new_file(directory/'plan.json',plan)
        sha=d.batch.base.wire.digest(directory/'plan.json')
        d.plan_binding(directory/'plan.json',sha)
        claim=claim_when_available(journal,journal_sha,pair['pair_id'],frozen,panel)
        if len(claim['attempts'])!=1 or claim['attempts'][0]['game']!=lookup[labels[index]] or claim['attempts'][0]['attempt_label']!=attempts[index]:
            raise ValueError('new claim differs from prepared game')
        rows.append({'pair_id':pair['pair_id'],'directory':str(directory),'plan_sha256':sha,'claim_token':claim['claim_token']})
        print(json.dumps({'prepared':len(rows),'total':len(pairs),'game':labels[index]}),flush=True)
    cohort={'schema':SCHEMA,'journal':str(journal),'journal_sha256':journal_sha,'max_workers':len(rows),'games':rows}
    q.new_file(output/'cohort.json',cohort)
    print(json.dumps({'cohort':str(output/'cohort.json'),'sha256':d.batch.base.wire.digest(output/'cohort.json')}),flush=True)
    return cohort


def run(path, sha, *, resume=False):
    path=Path(path).absolute()
    if d.batch.base.wire.digest(path)!=sha:raise ValueError('exact frozen cohort SHA required')
    c=q.read(path)
    if set(c)!={'schema','journal','journal_sha256','max_workers','games'} or c['schema']!=SCHEMA:
        raise ValueError('exact durable cohort required')
    if type(c['max_workers']) is not int or not 1<=len(c['games'])<=c['max_workers']<=64:raise ValueError('bounded64worker cohort required')
    if any(len({r[k] for r in c['games']})!=len(c['games']) for k in ('pair_id','directory','claim_token')):raise ValueError('duplicate claim')
    with (path.parent/'cohort.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=q.journal_path(c['journal']); frozen,panel=q.load_plan(journal,c['journal_sha256'])
        for row in c['games']:
            directory=Path(row['directory']);_,plan=d.plan_binding(directory/'plan.json',row['plan_sha256'])
            claim=q.find_claim(journal,row['claim_token']);q.validate_claim(claim,frozen,panel)
            index=plan['scope']['selected_game_indices'][0]
            if len(claim['attempts'])!=1 or claim['attempts'][0]['game']!=plan['games'][index] or claim['attempts'][0]['attempt_label']!=plan['artifact_labels'][index]:
                raise ValueError('exact single-game claim binding required')
            if not resume and (directory/'intent.json').exists():raise ValueError('existing intent requires explicit resume')
        def launch(row):
            directory=Path(row['directory'])
            if not (directory/'durable-result.json').exists():
                command=[sys.executable,str(d.batch.base.ROOT/d.ENTRY),'--run',str(directory/'plan.json'),'--sha',row['plan_sha256']]
                if resume and (directory/'submission.json').exists():command.append('--resume')
                with (directory/('resume.log' if resume else 'coordinator.log')).open('a' if resume else 'x') as log:
                    result=subprocess.run(command,cwd=d.batch.base.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
                if result.returncode:raise RuntimeError(f"{row['pair_id']}: retained cloud call requires recovery; no duplicate submission")
            return row
        outcomes=[]
        with ThreadPoolExecutor(max_workers=c['max_workers']) as pool:
            for future in as_completed([pool.submit(launch,r) for r in c['games']]):
                try:
                    row=future.result()
                    accepted=ingest_when_available(row,journal,c['journal_sha256'])
                    item={'pair_id':row['pair_id'],'status':'ingested','receipt':accepted}
                except Exception as error:item={'status':'requires-reconciliation','error':str(error)[:2048]}
                outcomes.append(item);print(json.dumps(item),flush=True)
        q.new_file(path.parent/('resume-result.json' if resume else 'cohort-result.json'),{'schema':SCHEMA,'outcomes':outcomes})
        return outcomes


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',type=Path);g.add_argument('--run',type=Path)
    p.add_argument('--journal',type=Path);p.add_argument('--journal-sha');p.add_argument('--sha')
    p.add_argument('--limit',type=int,default=64);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.prepare:prepare(a.prepare,a.journal,a.journal_sha,limit=a.limit)
    else:run(a.run,a.sha,resume=a.resume)
