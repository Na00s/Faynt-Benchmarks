"""Cloud fault-injection checks for durable orchestration. Runs zero games."""

import prepared_runtime
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import modal_panel_durable_game as d

ENTRY='scripts/modal_panel_durable_canary.py'


def fault_check(plan_sha, scenario, expires, image_id):
    import modal
    if scenario not in {'interrupt','after_commit'}:raise ValueError('exact canary required')
    bound,plan=d.plan_binding(d.batch.base.PLAN,plan_sha,remote=True)
    call=modal.current_function_call_id()
    sha=hashlib.sha256((plan_sha+scenario+call).encode()).hexdigest()
    volume=modal.Volume.from_name(d.VOLUME); store=modal.Dict.from_name(d.INDEX)
    execution={'task_id':os.environ['MODAL_TASK_ID'],'execution_id':uuid.uuid4().hex}
    class CommitFault:
        def reload(self):volume.reload()
        def commit(self):
            volume.commit()
            if scenario=='after_commit' and store.put(sha+'/commit-fault',True,skip_if_exists=True):
                raise RuntimeError('CANARY: crash after explicit durable commit and before return')
    def fake_owner():
        started={'execution':execution,'time':time.time()}
        first=store.put(sha+'/fixture-start',started,skip_if_exists=True)
        if scenario=='interrupt' and first:
            raise RuntimeError('CANARY: interrupted synthetic game before completion')
        if scenario=='after_commit' and not first:
            raise AssertionError('completed fixture was executed twice')
        body=json.dumps({'canary':scenario,'zero_real_games':True}).encode()
        digest=hashlib.sha256(body).hexdigest()
        return {'schema':bound.SCHEMA,'scope':{'canary':scenario,'benchmark_games':0},'status':d.SUCCESS,
            'files':[{'name':'game-0/result.json','body':body,'sha256':digest,
                      'original_bytes':len(body),'original_sha256':digest,'truncated_tail':False}]}
    manifest=d.run_durable(bound,plan,sha,expires,image_id,CommitFault(),store,call,execution,run_owner=fake_owner)
    first=store.get(sha+'/fixture-start')
    if first['execution']['task_id']==execution['task_id']:
        raise AssertionError('fault injection did not exercise replacement container')
    if manifest['physical_number'] != (2 if scenario=='interrupt' else 1):raise AssertionError('wrong execution count')
    return {'scenario':scenario,'passed':True,'benchmark_games':0,'manifest':manifest,
            'replacement_execution':execution,'original_execution':first['execution'],'call_id':call}


def run(path,sha,output):
    import modal
    bound,plan=d.plan_binding(path,sha)
    os.environ['MODAL_PROFILE']='frisson'
    app=modal.App('frisson-durable-recovery-canary-v1',include_source=False)
    volume=modal.Volume.from_name(d.VOLUME,create_if_missing=True,environment_name='main')
    store=modal.Dict.from_name(d.INDEX,create_if_missing=True,environment_name='main')
    image=modal.Image.from_registry(prepared_runtime.image_reference()).env({**prepared_runtime.image_environment(), 'PYTHONPATH':str(d.batch.base.REMOTE/'scripts')})
    for row in plan['uploads']:image=image.add_local_file(row['local'],row['remote'],copy=False)
    image=image.add_local_file(str(path),str(d.batch.base.PLAN),copy=False)
    image=image.add_local_file(str(d.batch.base.ROOT/ENTRY),str(d.batch.base.REMOTE/ENTRY),copy=False)
    fn=app.function(image=image,include_source=False,serialized=False,cpu=(4,4),memory=(16384,16384),
        max_containers=2,timeout=120,startup_timeout=120,nonpreemptible=True,single_use_containers=True,
        volumes={str(d.MOUNT):volume},retries=modal.Retries(max_retries=2,initial_delay=1.0))(fault_check)
    output.mkdir(parents=True,exist_ok=False)
    d.create_json(output/'intent.json',{'plan_sha256':sha,'benchmark_games':0,'scenarios':['interrupt','after_commit'],
                                      'canary_source_sha256':d.batch.base.wire.digest(d.batch.base.ROOT/ENTRY)})
    with modal.enable_output(),app.run(environment_name='main'):
        store.hydrate();d.create_json(output/'app.json',{'app_id':app.app_id,'image_id':image.object_id})
        expires=time.time()+240
        calls={name:fn.spawn(sha,name,expires,image.object_id) for name in ['interrupt','after_commit']}
        d.create_json(output/'calls.json',{n:c.object_id for n,c in calls.items()})
        results={}
        for name,call in calls.items():
            result=call.get()
            if result['call_id']!=call.object_id or not result['passed']:raise ValueError('actual fault canary failed')
            results[name]=result
            d.create_json(output/(name+'.json'),result)
        d.create_json(output/'result.json',{'passed':True,'benchmark_games':0,'results':results})
        return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.plan.absolute(),a.sha,a.output.absolute())))
