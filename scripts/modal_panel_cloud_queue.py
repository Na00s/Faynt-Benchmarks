"""E011 cloud queue state machine. No policy or emulator implementation changes.

The immutable migration snapshot contains every original game, including accepted
and quarantined rows. A deployed single dispatcher owns scheduling. Workers own
atomic call binding and durable native acceptance. Repeated dispatch after a
lost spawn response is safe because only the first bound call can run gameplay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 'e011.cloud-panel-queue.v1'
TERMINAL = frozenset({'complete', 'quarantined'})
MAX_WORKERS = 64

# Explicitly reviewed infrastructure failures. Each key binds the complete
# retained native manifest, including every file hash, original plan and call.
# The unchanged native manifest validator must pass before consulting this map.
# Evidence and raw-event audits are retained in cloud-dispatch-v1/failed-evidence
# and REVIEWED_FAILURE_RECOVERY.md. These exceptions authorize bounded retries;
# they never admit the failed attempt's observed outcome into the score ledger.
REVIEWED_FAILURE_RETRIES = {
    '1bd75c1e39ad9a01af887be731adf448b65aea67c53e8272f900b1ae1f378896':
        'reviewed-countdown-enet-disconnect',
    '4c364cb1a766bf06af98d69068d0c8145592b1f1a66cdc2d09e0e25f6b912034':
        'reviewed-truncated-terminal-replay',
    '818cd7acdcbaf7513cd9dd265e15ff2655bc50085584cd0401852999f41f371b':
        'reviewed-zero-gameplay-dependency-acquisition-failure',
    # Faynt D0 apt installation timeouts: exact six-file manifests, with zero
    # game starts, policy activity and ROM reads; recovery-audit/apt-timeouts.md.
    '39ac4fafb238b6a3e9a19624ee2f93b2735ba6dee88c6d74efe2820a7eea681b':
        'reviewed-zero-gameplay-dependency-acquisition-failure',
    'aa52e8ae30efe52d36cc336d2d01205822e0792fe856e2fffe682bcd707fde80':
        'reviewed-zero-gameplay-dependency-acquisition-failure',
    '5be643f443eec2f4bb964cd75ca6cbab000790df870c5bc85dc8b64c86f1e349':
        'reviewed-zero-gameplay-dependency-acquisition-failure',
    'e93411c06d3dff8fb37c8c5ef689d5ed86df00136b5c4b9badd9e21c8bd9c8dd':
        'reviewed-zero-gameplay-dependency-acquisition-failure',
    # Faynt D0 Mewtwo bootstrap deadline after two successful apt commands, with
    # zero gameplay; recovery-audit/fox24_mewtwo_fod/REVIEW.md.
    '659be2fb03c1a9cb583bd0628e65727166f91eac229e9e2d0cb24efa1cff77b4':
        'reviewed-zero-gameplay-runtime-bootstrap-deadline',
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else encoded(value)).hexdigest()


def budget_digest(value):
    """Use the budget ledger's canonical serialization for tranche identities."""
    from modal_panel_budget import digest as canonical_digest
    return canonical_digest(value)


def validate_snapshot(snapshot):
    if snapshot['schema'] != SCHEMA or snapshot['maximum_workers'] != MAX_WORKERS:
        raise ValueError('frozen64worker queue required')
    rows = snapshot['games']
    if len(rows) != 2624 or len({r['label'] for r in rows}) != len(rows):
        raise ValueError('complete2624game inventory required')
    manifest = snapshot['manifest']
    if digest(snapshot['manifest_body'].encode()) != snapshot['manifest_sha256']:
        raise ValueError('original manifest bytes differ')
    if json.loads(snapshot['manifest_body']) != manifest:
        raise ValueError('original manifest object differs')
    if [r['label'] for r in rows] != [g['label'] for g in manifest['games']]:
        raise ValueError('original ordered full inventory required')
    for row, game in zip(rows, manifest['games'], strict=True):
        if row['game_sha256'] != digest(game) or not game['save_slp'] or game['save_video']:
            raise ValueError('frozen SLP-only game differs')
        if row['initial_status'] not in TERMINAL | {'pending'}:
            raise ValueError('unresolved initial local status')
        if row['initial_status'] == 'pending':
            if not 0 <= row['phase'] < 6 or not 0 <= row['worker_slot'] < MAX_WORKERS:
                raise ValueError('original phase and slot required')
            if len(row['plan_sha256']) != 64 or row['plan_path'] != 'plans/'+row['label']+'.json':
                raise ValueError('bounded immutable game plan required')
    return snapshot


def status(row, records):
    return records.get(row['label'], {}).get('status', row['initial_status'])


def eligible(snapshot, records, *, limit=MAX_WORKERS, holds=(), strict_phase_order=False):
    """Preserve runnable slot order; strict mode keeps held phases open."""
    if type(strict_phase_order) is not bool:
        raise ValueError('strict_phase_order must be boolean')
    unfinished = [r for r in snapshot['games'] if status(r, records) not in TERMINAL]
    pending = [r for r in unfinished if r['label'] not in holds]
    phase_rows = unfinished if strict_phase_order else pending
    if not phase_rows: return []
    phase = min(r['phase'] for r in phase_rows)
    first = {}
    for row in pending:
        if row['phase'] == phase: first.setdefault(row['worker_slot'], row)
    if not strict_phase_order:
        # A lane with no runnable row in the current phase takes the earliest
        # runnable row of a later phase; current-phase rows keep lane priority.
        for row in pending:
            if row['phase'] > phase: first.setdefault(row['worker_slot'], row)
    return list(first.values())[:limit]


def bind_call(store, key, call_id, expires, image_id):
    """An ambiguous dispatch may be repeated; gameplay ownership stays unique."""
    value = {'call_id': call_id, 'expires_unix': expires, 'image_id': image_id}
    store.put(key, value, skip_if_exists=True)
    return store.get(key) == value


def execution_plan(row, metadata_body):
    """Retain the exact original byte identity when adopting a submitted plan.

    The first metadata package pretty-printed its retained JSON objects. Original
    cohort files used compact JSON. Their original SHA is still bound by the
    existing claim and submission. Reconstruct only that verified serialization;
    never change any value or substitute a different logical execution identity.
    """
    if digest(metadata_body)!=row['plan_sha256']:raise ValueError('queue plan bytes changed')
    plan=json.loads(metadata_body)
    if row.get('external_submission'):
        expected=row['existing_claim']['plan_sha256']
        if row['external_submission']['plan_sha256']!=expected:
            raise ValueError('original submitted plan and claim disagree')
        if digest(metadata_body)==expected:return plan,metadata_body,expected
        original=encoded(plan)
        if digest(original)!=expected:raise ValueError('original submitted plan bytes cannot be reconstructed')
        return plan,original,expected
    return plan,metadata_body,row['plan_sha256']


def execution_plan_sha(row):
    return row['existing_claim']['plan_sha256'] if row.get('external_submission') else row['plan_sha256']


def zero_runtime_timeout(manifest, root):
    """Only a proven pre-policy dependency-install timeout permits this retry."""
    names={r['name'] for r in manifest['files']}
    if manifest['status']!='failed' or any(n.startswith(('game-','project/','source-command/')) for n in names):return False
    try:
        owner=json.loads((root/'files/receipt.json').read_bytes())
        runtime=json.loads((root/'files/runtime/receipt.json').read_bytes())
        supervisor=json.loads((root/'files/supervisor/receipt.json').read_bytes())
        scope=runtime['scope'];last=runtime['commands'][-1]
        return (owner['status']=='failed' and owner['games']==[]
            and owner['fresh_runtime_status']=='failed' and owner['error']=='ValueError: fresh runtime failed'
            and runtime['status']=='failed' and runtime['error']=='TimeoutError: total build deadline exhausted'
            and scope['gameplay_admitted'] is False
            and all(scope[k]==0 for k in ('dolphin_launches','policy_inference','policy_instances','rom_reads'))
            and last['error']=='TimeoutError: total build deadline exhausted' and last['exit_code']==-15
            and last['terminated_owned_process_group'] is True and last['unreaped_leader_group_guard'] is True
            and 'pip' in last['command'] and 'install' in last['command']
            and supervisor['terminated_owned_process_group'] is True
            and supervisor['unreaped_leader_group_guard'] is True)
    except (OSError,KeyError,ValueError,IndexError):return False


def zero_menu_disconnect(manifest, root):
    """Recognize an ENet startup disconnect with no policy frames or replay."""
    try:
        rows=manifest['files'];names={r['name'] for r in rows}
        if manifest['status']!='failed' or any(n.endswith('.slp') for n in names):return False
        selected=manifest['scope']['selected_game_indices']
        if len(selected)!=1:return False
        idx=selected[0]
        summaries=[n for n in names if n.startswith('project/') and n.endswith('/summary.json')]
        if len(summaries)!=1:return False
        trace=summaries[0].removesuffix('summary.json')+'controller_trace.jsonl'
        traces=[r for r in rows if r['name'].endswith('/controller_trace.jsonl')]
        if len(traces)!=1 or traces[0]['name']!=trace or traces[0]['bytes']!=0:return False
        if any(n.startswith((f'game-{1-idx}/',f'game-command-{1-idx}/')) for n in names):return False
        def read(name):
            if name not in names:raise ValueError('missing retained proof')
            return json.loads((root/'files'/name).read_bytes())
        owner=read('receipt.json');child=read(f'game-{idx}/result.json')
        summary=read(summaries[0]);supervisor=read('supervisor/receipt.json')
        execution=summary['execution'];games=owner['games']
        if not (owner['status']=='failed' and owner['error']=='ValueError: native game acceptance failed'
                and len(games)==1 and games[0]['index']==idx
                and games[0]['child']==child
                and games[0]['cleanup']['parent_terminal_cleanup_verified'] is True
                and child['status']=='failed' and child['error']=='RuntimeError: EnetDisconnected: '
                and summary['error']=='EnetDisconnected: '
                and execution['processed_policy_frames']==0
                and execution['first_game_frame'] is None and execution['last_game_frame'] is None
                and (root/'files'/trace).stat().st_size==0
                and supervisor['exit_code']==0
                and supervisor['terminated_owned_process_group'] is True
                and supervisor['unreaped_leader_group_guard'] is True):return False
        for port in ('p1','p2'):
            diagnostics=summary['policies'][port]['diagnostics']
            if not (diagnostics['frames_total']==0 and diagnostics['frames_in_generation']==0
                    and diagnostics['current_frame_inference_barriers']==0
                    and diagnostics['first_frame'] is None and diagnostics['last_frame'] is None):return False
        return True
    except (OSError,KeyError,ValueError,IndexError,TypeError):return False


def zero_gameplay_retry_reason(manifest, root):
    if zero_runtime_timeout(manifest,root):return 'verified-zero-gameplay-runtime-install-timeout'
    if zero_menu_disconnect(manifest,root):return 'verified-zero-policy-frame-enet-startup-disconnect'
    return None


def interrupted_native_owner(manifest, root):
    """Identify a cleaned-up interrupted owner, retaining every partial replay.

    This grants another bounded execution only. It never establishes a game
    outcome. The caller must validate the complete manifest's file hashes first.
    A terminal summary or child result requires its own scientific review.
    """
    try:
        names={row['name'] for row in manifest['files']}
        selected=manifest['scope']['selected_game_indices']
        if (manifest['status']!='failed' or selected not in ([0],[1])
                or any(name.endswith('/summary.json') for name in names)
                or any(name in names for name in ('game-0/result.json','game-1/result.json'))
                or not {'receipt.json','supervisor/receipt.json'}<=names):
            return False
        owner=json.loads((root/'files/receipt.json').read_bytes())
        supervisor=json.loads((root/'files/supervisor/receipt.json').read_bytes())
        return (owner['status']=='failed' and owner['games']==[]
            and owner['error']=='KeyboardInterrupt: owner termination requested'
            and owner['plan_sha256']==manifest['plan_sha256']
            and owner['scope']['selected_game_indices']==selected
            and supervisor['error']=='KeyboardInterrupt: '
            and supervisor['exit_code']==0
            and supervisor['terminated_owned_process_group'] is True
            and supervisor['unreaped_leader_group_guard'] is True)
    except (OSError,KeyError,ValueError,TypeError):
        return False


def interrupted_empty_owner(manifest, root):
    """Recognize the reviewed early SIGTERM with only empty supervisor output.

    The exact two-file inventory excludes completed outcome artifacts. It grants
    a bounded retry and makes no claim about an observed gameplay result.
    """
    import math
    import re
    try:
        rows=manifest['files']
        names={row['name'] for row in rows}
        if (manifest['status']!='failed' or len(rows)!=2
                or names!={'supervisor/owner.log','supervisor/receipt.json'}
                or manifest['scope']['selected_game_indices'] not in ([0],[1])):
            return False
        log=next(row for row in rows if row['name']=='supervisor/owner.log')
        if (log['bytes']!=0 or log['original_bytes']!=0 or log['truncated_tail'] is not False
                or log['sha256']!=digest(b'') or log['original_sha256']!=digest(b'')
                or (root/'files/supervisor/owner.log').read_bytes()!=b''):
            return False
        supervisor=json.loads((root/'files/supervisor/receipt.json').read_bytes())
        command=supervisor['command']
        if (not isinstance(command,list) or len(command)!=9
                or command[1]!='/opt/d0-root/scripts/faynt_d0_durable_game.py'
                or command[2:5]!=['--owner','--sha',manifest['plan_sha256']]
                or command[5]!='--expires' or command[7]!='--image-id'
                or not re.fullmatch(r'im-[A-Za-z0-9]+',command[8])):
            return False
        start=supervisor['started_unix'];end=supervisor['ended_unix'];expiry=supervisor['expires_unix']
        if (any(type(value) not in (int,float) or not math.isfinite(value)
                for value in (start,end,expiry))
                or not 0<start<=end<expiry or end-start>10
                or float(command[6])!=expiry):
            return False
        return (supervisor['error']=='KeyboardInterrupt: '
            and supervisor['exit_code']==-15
            and supervisor['terminated_owned_process_group'] is True
            and supervisor['unreaped_leader_group_guard'] is True
            and supervisor['log_bytes']==0 and supervisor['log_sha256']==digest(b''))
    except (OSError,KeyError,ValueError,TypeError,IndexError):
        return False


def exhausted_prebootstrap_budget(manifest, root):
    """Recognize an owner rejected before its reserved bootstrap could begin."""
    import math
    try:
        rows=manifest['files'];names={row['name'] for row in rows}
        if (manifest['status']!='failed' or len(rows)!=3
                or names!={'receipt.json','supervisor/owner.log','supervisor/receipt.json'}):
            return False
        owner=json.loads((root/'files/receipt.json').read_bytes())
        supervisor=json.loads((root/'files/supervisor/receipt.json').read_bytes())
        log=next(row for row in rows if row['name']=='supervisor/owner.log')
        start=owner['started_unix'];end=owner['ended_unix'];expiry=owner['expires_unix']
        if (any(type(value) not in (int,float) or not math.isfinite(value)
                for value in (start,end,expiry,supervisor['started_unix'],supervisor['ended_unix']))
                or not 0<supervisor['started_unix']<=start<=end<=supervisor['ended_unix']<expiry
                or end-start>1 or not 0<expiry-start<=2100):
            return False
        command=supervisor['command']
        return (owner['status']=='failed' and owner['games']==[]
            and owner['error']=='TimeoutError: pilot absolute deadline exhausted'
            and 'fresh_runtime_status' not in owner
            and owner['plan_sha256']==manifest['plan_sha256']
            and owner['scope']==manifest['scope']
            and owner['scope']['selected_game_indices'] in ([0],[1])
            and supervisor['expires_unix']==expiry and 'error' not in supervisor
            and isinstance(command,list) and len(command)==9
            and command[1]=='/opt/d0-root/scripts/faynt_d0_durable_game.py'
            and command[2:5]==['--owner','--sha',manifest['plan_sha256']]
            and command[5]=='--expires' and float(command[6])==expiry
            and command[7]=='--image-id' and command[8]==owner['runtime_image_id']
            and supervisor['exit_code']==0
            and supervisor['terminated_owned_process_group'] is True
            and supervisor['unreaped_leader_group_guard'] is True
            and supervisor['log_bytes']==0 and supervisor['log_sha256']==digest(b'')
            and log['bytes']==0 and log['original_bytes']==0 and log['truncated_tail'] is False
            and log['sha256']==digest(b'') and log['original_sha256']==digest(b'')
            and (root/'files/supervisor/owner.log').read_bytes()==b'')
    except (OSError,KeyError,ValueError,TypeError,IndexError):
        return False


def zero_gameplay_public_source_fetch(manifest, root):
    """Match the reviewed public Git fetch failure before any game was started."""
    import re
    try:
        rows=manifest['files'];names={row['name'] for row in rows}
        required={'receipt.json','runtime/receipt.json','runtime/qualification.json',
            'source-command/commands.json','supervisor/receipt.json','supervisor/owner.log',
            *(f'source-command/logs/{number:03d}.log' for number in range(1,5))}
        if (manifest['status']!='failed' or len(names)!=len(rows) or not required<=names
                or any(not re.fullmatch(r'(?:receipt\.json|runtime/(?:receipt\.json|qualification\.json|'
                    r'logs/[0-9]{3}\.log)|source-command/(?:commands\.json|logs/00[1-4]\.log)|'
                    r'supervisor/(?:receipt\.json|owner\.log))',name) for name in names)):
            return False
        def read(name):
            return json.loads((root/'files'/name).read_bytes())
        owner=read('receipt.json');runtime=read('runtime/receipt.json')
        qualified=read('runtime/qualification.json');supervisor=read('supervisor/receipt.json')
        commands=read('source-command/commands.json');scope=runtime['scope']
        if not (owner['status']=='failed' and owner['games']==[]
                and owner['error']=='RuntimeError: command exited 128; inspect 004.log'
                and owner['fresh_runtime_status']=='runtime-qualified-gameplay-unadmitted'
                and owner['plan_sha256']==manifest['plan_sha256'] and owner['scope']==manifest['scope']
                and owner['scope']['selected_game_indices'] in ([0],[1])
                and runtime['status']=='runtime-qualified-gameplay-unadmitted' and 'error' not in runtime
                and scope['gameplay_admitted'] is False
                and all(type(scope[key]) is int and scope[key]==0
                    for key in ('dolphin_launches','rom_reads','policy_instances','policy_inference'))
                and runtime['qualification']==qualified and qualified['scope']==scope
                and qualified['status']=='runtime-child-qualified'
                and qualified['parent_terminal_cleanup_verified'] is False
                and runtime['cleanup']['parent_terminal_cleanup_verified'] is True
                and supervisor['exit_code']==0 and 'error' not in supervisor
                and supervisor['terminated_owned_process_group'] is True
                and supervisor['unreaped_leader_group_guard'] is True
                and supervisor['expires_unix']==owner['expires_unix']):
            return False
        supervisor_command=supervisor['command']
        if (not isinstance(supervisor_command,list) or len(supervisor_command)!=9
                or supervisor_command[1]!='/opt/d0-root/scripts/faynt_d0_durable_game.py'
                or supervisor_command[2:5]!=['--owner','--sha',manifest['plan_sha256']]
                or supervisor_command[5]!='--expires'
                or float(supervisor_command[6])!=owner['expires_unix']
                or supervisor_command[7]!='--image-id'
                or supervisor_command[8]!=owner['runtime_image_id']):
            return False
        work='/work/research/faynt-d0-game'
        public=work+'/source/melee-policy/.e001-cache/slippi-ai-source'
        expected=[
            ['apt-get','-o','Acquire::Retries=0','-o','Acquire::http::Timeout=30',
                'install','-y','--no-install-recommends','git'],
            ['git','init','--initial-branch=main',public],
            ['git','-C',public,'remote','add','origin','https://github.com/vladfi1/slippi-ai'],
            ['git','-C',public,'fetch','--depth','1','--no-tags','origin',
                '577965a7731dc53e3472ea63d9e9853a4e9d65fa'],
        ]
        if not isinstance(commands,list) or len(commands)!=4:
            return False
        for number,(command,argv) in enumerate(zip(commands,expected,strict=True),1):
            log_name=f'source-command/logs/{number:03d}.log'
            log=next(row for row in rows if row['name']==log_name)
            if (command['command']!=argv or command['cwd']!=work
                    or command['log']!=f'logs/{number:03d}.log'
                    or command['exit_code']!=(128 if number==4 else 0)
                    or command['terminated_owned_process_group'] is not True
                    or command['unreaped_leader_group_guard'] is not True
                    or command['log_bytes']!=log['original_bytes']
                    or command['log_sha256']!=log['original_sha256']
                    or (number<4 and 'error' in command)):
                return False
        body=(root/'files/source-command/logs/004.log').read_bytes()
        return (commands[-1]['error']=='RuntimeError: command exited 128; inspect 004.log'
            and len(body)==126
            and digest(body)=='a3ed1bca695f5d110bed3bb470877f16707bd532b22040cb10393b8033300f4c')
    except (OSError,KeyError,ValueError,TypeError,IndexError,StopIteration):
        return False


def verified_failure_retry_reason(manifest, root):
    """Called after strict native artifact validation, with exact attempt proofs."""
    reason=zero_gameplay_retry_reason(manifest,root)
    if reason:return reason
    if interrupted_native_owner(manifest,root):return 'verified-interrupted-native-owner'
    if interrupted_empty_owner(manifest,root):return 'verified-preowner-interruption'
    if exhausted_prebootstrap_budget(manifest,root):return 'verified-prebootstrap-budget-exhaustion'
    if zero_gameplay_public_source_fetch(manifest,root):return 'verified-zero-gameplay-public-source-fetch'
    if manifest['status']!='failed':return None
    return REVIEWED_FAILURE_RETRIES.get(digest(manifest))


def retry_execution_plan(row, metadata_body, authorization):
    import copy
    original,_,original_sha=execution_plan(row,metadata_body)
    if (authorization['queue_plan_sha256']!=row['plan_sha256']
            or authorization['original_execution_plan_sha256']!=original_sha
            or authorization['queue_plan_body']!=metadata_body.decode()
            or authorization['reason'] not in ('verified-zero-gameplay-runtime-install-timeout',
                'verified-zero-policy-frame-enet-startup-disconnect',
                'verified-interrupted-native-owner',
                'verified-preowner-interruption',
                'verified-prebootstrap-budget-exhaustion',
                'verified-zero-gameplay-public-source-fetch',
                *REVIEWED_FAILURE_RETRIES.values())):
        raise ValueError('retry authorization does not bind original frozen inputs')
    number=authorization['attempt_number']
    if type(number) is not int or not 2<=number<=3:raise ValueError('three-attempt retry bound')
    histories=[h for failure in authorization['prior_failures'] for h in failure['physical_history']]
    if len(histories)!=number-1 or len({h['execution']['execution_id'] for h in histories})!=len(histories):
        raise ValueError('every prior physical attempt must be retained once')
    plan=copy.deepcopy(original);idx=plan['scope']['selected_game_indices'][0]
    plan['artifact_labels'][idx]=row['label']+f'-a{number}'
    body=encoded(plan);sha=digest(body)
    if sha!=authorization['plan_sha256']:raise ValueError('retry changed more than the native attempt label')
    return plan,body,sha


def authorize_failure_retry(row, metadata_body, prior_failures, *, reason,
                            expires_unix, now, session_seconds,
                            resume_tranche=None, accepted_record=None,
                            existing_authorization=None):
    """Build a label-only retry, retaining original calls and physical starts.

    A validated immutable budget tranche is the sole route to a later deadline.
    The cloud application verifies that tranche against the stopped application
    proof and its current budget selector before calling this helper.
    """
    import copy
    import math
    if row.get('initial_status','pending') in TERMINAL or (
            accepted_record and accepted_record.get('status') in TERMINAL):
        raise ValueError('terminal games cannot be retried')
    if any(type(value) not in (int,float) or not math.isfinite(value)
           for value in (now,expires_unix,session_seconds)) or session_seconds<=600:
        raise ValueError('finite bounded retry timing required')
    original,_,original_sha=execution_plan(row,metadata_body)
    number=1+sum(len(item['physical_history']) for item in prior_failures)
    if not 2<=number<=3:
        raise ValueError('three-attempt retry bound')
    if any(item['manifest']['status']!='failed' for item in prior_failures):
        raise ValueError('successful or unresolved prior execution cannot be retried')
    if reason=='verified-prebootstrap-budget-exhaustion' and resume_tranche is None:
        raise ValueError('prebootstrap budget recovery requires an approved current tranche')
    if existing_authorization is not None:
        existing=existing_authorization
        if (existing.get('reason')!=reason or existing.get('attempt_number')!=number
                or existing.get('prior_failures')!=prior_failures
                or existing.get('original_expires_unix',existing.get('expires_unix'))!=expires_unix):
            raise ValueError('existing retry authorization differs from retained failure proofs')
        retry_execution_plan(row,metadata_body,existing)
        validate_retry_budget_binding(existing,resume_tranche)
        # Publication may have succeeded before retry-latest or the response.
        # Preserve the first frozen deadline, including when it has since expired.
        return copy.deepcopy(existing)
    expiry=expires_unix
    budget_binding=None
    if now+600>=expiry:
        if resume_tranche is None:
            raise ValueError('original deadline exhausted; explicit budget resume required')
        deadline=resume_tranche['activation']['deadline_unix']
        if type(deadline) not in (int,float) or not math.isfinite(deadline):
            raise ValueError('finite resumed budget deadline required')
        expiry=min(now+session_seconds,deadline)
        if now+600>=expiry:
            raise ValueError('resumed tranche has insufficient execution budget')
        budget_binding={'tranche_sha256':budget_digest(resume_tranche),
            'authorization_sha256':resume_tranche['authorization_sha256'],
            'activation_sha256':budget_digest(resume_tranche['activation'])}
    replacement=copy.deepcopy(original)
    replacement['artifact_labels'][replacement['scope']['selected_game_indices'][0]]=row['label']+f'-a{number}'
    authorization={'reason':reason,'attempt_number':number,
        'queue_plan_sha256':row['plan_sha256'],'original_execution_plan_sha256':original_sha,
        'queue_plan_body':metadata_body.decode(),'plan_sha256':digest(encoded(replacement)),
        'expires_unix':expiry,'original_expires_unix':expires_unix,
        'prior_failures':copy.deepcopy(prior_failures)}
    if budget_binding is not None:
        authorization['budget_resume']=budget_binding
    retry_execution_plan(row,metadata_body,authorization)
    return authorization


def validate_retry_budget_binding(authorization, active_tranche):
    """A resumed deadline stays bound to the currently approved tranche."""
    resume=authorization.get('budget_resume')
    if resume is None:
        if authorization.get('original_expires_unix',authorization['expires_unix'])!=authorization['expires_unix']:
            raise ValueError('changed retry deadline lacks a budget resume')
        return
    if (active_tranche is None or resume!={
            'tranche_sha256':budget_digest(active_tranche),
            'authorization_sha256':active_tranche['authorization_sha256'],
            'activation_sha256':budget_digest(active_tranche['activation'])}
            or authorization['expires_unix']>active_tranche['activation']['deadline_unix']):
        raise ValueError('retry deadline requires the current hash-bound budget tranche')


# Fields a renewal may never touch. Only the deadline, its tranche binding and
# the renewal lineage change; the attempt bound and plan identity are inherited.
RENEWAL_FROZEN_FIELDS=('reason','attempt_number','queue_plan_sha256','original_execution_plan_sha256',
    'queue_plan_body','plan_sha256','original_expires_unix','prior_failures')


def validate_retry_renewal(expired, renewed):
    """A renewed authorization retains every frozen field of the record it replaces."""
    import math
    if not isinstance(expired,dict) or not isinstance(renewed,dict):
        raise ValueError('retained expired and renewed retry records required')
    if set(renewed)!=set(expired)|{'budget_resume','renews_authorization_sha256','renewal_count'}:
        raise ValueError('retry renewal changed the retained record shape')
    if any(field not in expired or encoded(renewed[field])!=encoded(expired[field])
           for field in RENEWAL_FROZEN_FIELDS):
        raise ValueError('retry renewal changed a frozen field')
    if renewed['renews_authorization_sha256']!=digest(expired):
        raise ValueError('retry renewal must bind the retained expired record')
    count=expired.get('renewal_count',0)
    if type(count) is not int or count<0 or renewed['renewal_count']!=count+1:
        raise ValueError('retry renewal count must advance by one')
    if (any(type(record['expires_unix']) not in (int,float) or not math.isfinite(record['expires_unix'])
            for record in (expired,renewed)) or renewed['expires_unix']<=expired['expires_unix']):
        raise ValueError('retry renewal must extend past the expired deadline')


def renew_expired_retry(row, metadata_body, expired, *, now, session_seconds, resume_tranche, index_has_start):
    """Re-time an expired authorization with zero physical starts under the active tranche.

    A started attempt keeps its frozen deadline forever; the caller proves the
    durable index holds no call binding and no physical slot for this attempt
    plan immediately before writing. Every frozen field stays byte-identical.
    Only expires_unix and budget_resume change, and the replaced record is
    bound by digest so its archived copy can be verified later.
    """
    import copy
    import math
    if type(index_has_start) is not bool:
        raise ValueError('explicit durable index inspection required')
    if index_has_start:
        raise ValueError('started retry attempt keeps its frozen deadline')
    if any(type(value) not in (int,float) or not math.isfinite(value)
           for value in (now,session_seconds)) or session_seconds<=600:
        raise ValueError('finite bounded retry timing required')
    if (not isinstance(expired,dict)
            or any(type(expired.get(name)) not in (int,float) or not math.isfinite(expired[name])
                   for name in ('expires_unix','original_expires_unix'))):
        raise ValueError('published retry authorization with frozen deadlines required')
    count=expired.get('renewal_count',0)
    if type(count) is not int or count<0:
        raise ValueError('retained renewal count required')
    retry_execution_plan(row,metadata_body,expired)
    if now<=expired['expires_unix']:
        raise ValueError('unexpired retry keeps its frozen deadline')
    if resume_tranche is None:
        raise ValueError('retry renewal requires an approved current tranche')
    deadline=resume_tranche['activation']['deadline_unix']
    if type(deadline) not in (int,float) or not math.isfinite(deadline):
        raise ValueError('finite resumed budget deadline required')
    expiry=min(now+session_seconds,deadline)
    if now+600>=expiry:
        raise ValueError('resumed tranche has insufficient execution budget')
    renewed={**copy.deepcopy(expired),'expires_unix':expiry,
        'budget_resume':{'tranche_sha256':budget_digest(resume_tranche),
            'authorization_sha256':resume_tranche['authorization_sha256'],
            'activation_sha256':budget_digest(resume_tranche['activation'])},
        'renews_authorization_sha256':digest(expired),'renewal_count':count+1}
    validate_retry_renewal(expired,renewed)
    validate_retry_budget_binding(renewed,resume_tranche)
    retry_execution_plan(row,metadata_body,renewed)
    return renewed


def validate_prior_call_binding(failure, actual_binding, authorization):
    """Keep each prior expiry unchanged when a new tranche resumes execution."""
    expected=failure.get('call_binding')
    if expected is None:
        # Legacy failure receipts predate explicit call_binding. Their frozen
        # physical history retains the old deadline even after later attempts
        # receive a new allocation. The caller also checks that history against
        # the native index and validates the prior manifest independently.
        history=failure.get('physical_history',[])
        if (actual_binding is None or actual_binding['call_id']!=failure['manifest']['call_id']
                or not history or any(start['call_id']!=actual_binding['call_id']
                    or start['expires_unix']!=actual_binding['expires_unix'] for start in history)):
            raise ValueError('prior call or original deadline changed')
        return
    history=failure['physical_history']
    if (actual_binding!=expected or expected['call_id']!=failure['manifest']['call_id']
            or not history or any(start['call_id']!=expected['call_id']
                or start['expires_unix']!=expected['expires_unix'] for start in history)):
        raise ValueError('retained prior call or deadline changed')


def held_reinspection_labels(control, active_tranche, definitions, records):
    """Explicitly scope one read-only hold reinspection per approved tranche."""
    request=control.get('held_reinspection')
    if request is None:return frozenset()
    if (not isinstance(request,dict) or set(request) not in (
            {'tranche_sha256','labels'},{'tranche_sha256','labels','revisions'})
            or active_tranche is None or request['tranche_sha256']!=budget_digest(active_tranche)):
        raise ValueError('held reinspection requires the current hash-bound budget tranche')
    labels=request['labels']
    if (not isinstance(labels,list) or not labels or len(labels)>2624
            or any(not isinstance(label,str) for label in labels)
            or len(labels)!=len(set(labels)) or any(label not in definitions for label in labels)):
        raise ValueError('explicit unique original held-game labels required')
    allowed=control.get('allowlist')
    if allowed is not None and not set(labels)<=set(allowed):
        raise ValueError('held reinspection must stay inside the active allowlist')
    if 'revisions' in request:
        import re
        revisions=request['revisions']
        if (not isinstance(revisions,dict) or not 1<=len(revisions)<=2
                or not set(revisions)<=set(labels)):
            raise ValueError('reviewed classifier revision is limited to two named held games')
        source_sha=digest(Path(__file__).read_bytes())
        for revision in revisions.values():
            if (not isinstance(revision,dict) or set(revision)!={'classifier_sha256','prior_hold_sha256'}
                    or revision['classifier_sha256']!=source_sha
                    or not isinstance(revision['prior_hold_sha256'],str)
                    or not re.fullmatch(r'[0-9a-f]{64}',revision['prior_hold_sha256'])):
                raise ValueError('reviewed classifier revision requires exact deployed code and prior hold hashes')
    return frozenset(label for label in labels if status(definitions[label],records) not in TERMINAL)


def held_reinspection_token(control, label, tranche_sha, hold):
    """A reviewed code revision gets its own bounded read-only recovery counter."""
    revision=control.get('held_reinspection',{}).get('revisions',{}).get(label)
    if revision is None:return tranche_sha
    token=digest({'tranche_sha256':tranche_sha,'label':label,**revision})
    if (hold is None or (digest(hold)!=revision['prior_hold_sha256']
            and hold.get('reinspection_token')!=token)):
        raise ValueError('reviewed classifier revision prior hold changed')
    return token


def receipt_plan_sha(row,receipt):
    retry=receipt.get('retry_authorization')
    if retry:
        _,_,sha=retry_execution_plan(row,retry['queue_plan_body'].encode(),retry)
        return sha
    return execution_plan_sha(row)


def validate_exclusion(row, receipt, queue_sha):
    """An explicit unscored terminal record retains all three failed starts."""
    if (row['initial_status']!='pending'
            or receipt.get('schema')!=SCHEMA+'.infrastructure-exclusion'
            or receipt.get('status')!='quarantined'
            or receipt.get('label')!=row['label']
            or receipt.get('queue_sha256')!=queue_sha
            or receipt.get('queue_plan_sha256')!=row['plan_sha256']
            or not receipt.get('user_authorization')
            or receipt.get('reason')!='infrastructure-failure-three-starts-exhausted'
            or 'native_acceptance' in receipt or 'result' in receipt):
        raise ValueError('explicit unscored exclusion binding required')
    try:
        retry=receipt['retry_authorization'];metadata=retry['queue_plan_body'].encode()
        _,_,retry_sha=retry_execution_plan(row,metadata,retry)
        failures=receipt['failed_attempts']
        if not isinstance(failures,list) or len(failures)!=3:
            raise ValueError('three retained failed starts required')
        lineage=[]
        for index,prior in enumerate(retry['prior_failures']):
            _,_,prior_sha=prior_execution_plan(row,metadata,retry,index)
            history=prior['physical_history']
            if not isinstance(history,list) or not history or len(lineage)+len(history)>=3:
                raise ValueError('bounded prior physical history required')
            last=len(lineage)+len(history)-1
            if failures[last]['manifest']!=prior['manifest']:
                raise ValueError('published prior failure manifest changed')
            lineage.extend((prior_sha,start,prior) for start in history)
        if len(lineage)!=retry['attempt_number']-1:
            raise ValueError('retry start position differs from published history')
        lineage.extend((retry_sha,None,None) for _ in range(3-len(lineage)))
        seen=set();calls={};bindings={}
        for number,(failure,(expected,retained_start,prior)) in enumerate(zip(failures,lineage),1):
            manifest=failure['manifest'];start=failure['physical_start']
            identity=manifest['execution']['execution_id']
            if (manifest['status']!='failed' or manifest['plan_sha256']!=expected
                    or type(manifest['physical_number']) is not int or manifest['physical_number']!=number
                    or identity in seen or manifest['execution']!=start['execution']
                    or manifest['call_id']!=start['call_id']
                    or calls.setdefault(expected,start['call_id'])!=start['call_id']):
                raise ValueError('failed physical provenance differs')
            binding=failure.get('call_binding')
            if prior is not None:
                if start!=retained_start:
                    raise ValueError('published prior physical start changed')
                # Older publications have no call_binding. Their unchanged
                # history and original expiry still establish the logical call.
                if binding is None:
                    binding=prior.get('call_binding') or {
                        'call_id':start['call_id'],'expires_unix':start['expires_unix']}
                validate_prior_call_binding(prior,binding,retry)
            elif start['expires_unix']!=retry['expires_unix']:
                raise ValueError('latest retry deadline changed')
            if binding is not None and (binding['call_id']!=start['call_id']
                    or binding['expires_unix']!=start['expires_unix']):
                raise ValueError('failed physical call binding differs')
            supplied=failure.get('call_binding')
            if supplied is not None and bindings.setdefault(expected,supplied)!=supplied:
                raise ValueError('one native plan requires one retained call binding')
            seen.add(identity)
    except (KeyError,TypeError,IndexError,AttributeError) as error:
        raise ValueError('complete retained physical provenance required') from error
    return receipt


def prior_execution_plan(row, metadata_body, authorization, failure_index):
    """Reconstruct the exact old attempt plan for its own strict validator."""
    retry_execution_plan(row,metadata_body,authorization)
    failures=authorization['prior_failures']
    if type(failure_index) is not int or not 0<=failure_index<len(failures):
        raise ValueError('exact prior failure index required')
    plan,body,sha=execution_plan(row,metadata_body)
    if failure_index:
        number=1+sum(len(f['physical_history']) for f in failures[:failure_index])
        plan['artifact_labels'][plan['scope']['selected_game_indices'][0]]=row['label']+f'-a{number}'
        body=encoded(plan);sha=digest(body)
    if failures[failure_index]['manifest']['plan_sha256']!=sha:
        raise ValueError('prior failure changed its original attempt plan')
    return plan,body,sha


def checked_materialize(row, blob_root):
    """Copy hash-bound inputs into their exact approved absolute container paths."""
    import os
    import shutil
    roots = blob_root if isinstance(blob_root, (list, tuple)) else [blob_root]
    sources = [Path(root)/row['sha256'] for root in roots if (Path(root)/row['sha256']).is_file()]
    if len(sources) != 1: raise ValueError('unique input blob required')
    source = sources[0]
    if source.stat().st_size != row['bytes']:
        raise ValueError('cloud input size differs')
    h = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): h.update(chunk)
    if h.hexdigest() != row['sha256']: raise ValueError('cloud input hash differs')
    for name in dict.fromkeys((row['remote'], row['local'])):
        target = Path(name)
        local_root = os.environ.get('FAYNT_BENCHMARK_LOCAL_ROOT')
        allowed = [Path('/opt')]
        if local_root:
            root = Path(local_root)
            if not root.is_absolute() or root == Path('/') or '..' in root.parts:
                raise ValueError('canonical benchmark local root required')
            allowed.append(root)
        if not any(target.is_relative_to(root) for root in allowed) or '..' in target.parts:
            raise ValueError('unapproved materialization target')
        if target.exists():
            if target.is_symlink() or target.stat().st_size != row['bytes']:
                raise ValueError('existing input differs')
            with target.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != row['sha256']: raise ValueError('existing input hash differs')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8*1024**2)
            dst.flush(); os.fsync(dst.fileno())


def native_acceptance(d, bound, plan, manifest):
    """Repeat the full existing terminal validator against committed cloud bytes."""
    # Modal supplies a symlink at the volume mount root. The original validator
    # requires canonical file paths; resolve the platform root while retaining
    # check_manifest's exact containment and per-file no-symlink checks.
    root = (d.MOUNT / manifest['storage_path']).resolve()
    d.check_manifest(manifest, manifest['plan_sha256'], root, bound)
    native = {**manifest, 'schema': manifest['native_schema']}
    files = {r['name']: root/'files'/r['name'] for r in manifest['files']}
    if manifest['status'] != d.SUCCESS:
        raise ValueError('failed native game is retained without score admission')
    bound.validate_terminal(native, files, manifest['plan_sha256'], plan)
    index = plan['scope']['selected_game_indices'][0]
    other = plan['artifact_labels'][1-index]
    if any(n.startswith((f'game-{1-index}/', f'game-command-{1-index}/',
                       f'project/artifacts/integration/frisson_ai/{other}/')) for n in files):
        raise ValueError('unselected peer artifacts')
    child = json.loads(files[f'game-{index}/result.json'].read_bytes())
    return child['native_acceptance']
