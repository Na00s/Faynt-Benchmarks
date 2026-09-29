"""Execute only game1 of an original pair after game0 was already retained.

Native gameplay and all per-game acceptance gates are reused unchanged. The
reviewed parent-loop changes select index1 and expect exactly one child result.
"""
from __future__ import annotations
import argparse
import ast
import inspect
from pathlib import Path
import textwrap
import types

import modal_panel_batch_pilot as batch

ENTRY = "scripts/modal_panel_single_pilot.py"
SUCCESS = "full-policy-selected-game-passed"


def select_function(function, namespace, *, owner=False):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    class Selection(ast.NodeTransformer):
        def __init__(self): self.loops=0; self.lengths=0; self.statuses=0
        def visit_For(self, node):
            self.generic_visit(node)
            wanted = "(0, 1)" if owner else "enumerate(result['games'])"
            if ast.unparse(node.iter) == wanted:
                node.iter = ast.parse("(1,)" if owner else "((1, result['games'][0]),)", mode="eval").body
                self.loops += 1
            return node
        def visit_Compare(self, node):
            self.generic_visit(node)
            if not owner and ast.unparse(node) == "len(result.get('games', [])) != 2":
                node.comparators[0] = ast.Constant(1); self.lengths += 1
            return node
        def visit_Constant(self,node):
            if node.value == "full-policy-compatibility-pair-passed":
                self.statuses += 1; return ast.copy_location(ast.Constant(SUCCESS), node)
            return node
    change=Selection(); tree=change.visit(tree)
    if change.loops != 1 or change.lengths != (0 if owner else 1) or change.statuses != 1:
        raise ValueError("proven selected-game orchestration structure changed")
    exec(compile(ast.fix_missing_locations(tree), "<selected-game-orchestration>", "exec"), namespace)
    return namespace[function.__name__]


def bind(labels, attempt_labels, *, remote=False):
    root = batch.base.REMOTE if remote else batch.base.ROOT
    panel = batch.base.policy.bounded_json(root / batch.base.PANEL)
    original = batch.binding.bind(panel, labels, attempt_labels, project_root=root,
        entry_relative="scripts/modal_panel_batch_pilot.py")
    ns = original.owner.__globals__
    scope = original.fixed_scope()
    scope.update(benchmark_games=1, selected_game_indices=[1], retained_pair_context=True)
    ns.update(ENTRY=batch.base.REMOTE/ENTRY,
        SOURCE_PATHS=(*original.SOURCE_PATHS, ENTRY), fixed_scope=lambda: dict(scope))
    select_function(original.owner, ns, owner=True)
    select_function(original.validate_terminal, ns)
    bound=types.SimpleNamespace(**ns)
    bound.binding={**original.binding, "scope":scope, "selected_game_indices":[1],
        "native_game_code_changed":False}
    return bound


def plan_binding(path, sha, *, remote=False):
    if batch.base.wire.digest(path) != sha: raise ValueError("exact selected-game plan required")
    plan=batch.base.policy.bounded_json(path)
    bound=bind([g["label"] for g in plan["games"]], plan["artifact_labels"], remote=remote)
    bound.verify_plan(path,sha,remote=remote)
    return bound,plan


def remote_single(sha,expires,image_id):
    bound,_=plan_binding(batch.base.PLAN,sha,remote=True)
    yield from bound.remote_game(sha,expires,image_id)


def configure_app(modal,path,plan,bound):
    # Use the same reviewed deployment resources and source upload mechanism,
    # with the importable selected-game remote function as the only change.
    ns=dict(batch.configure_app.__globals__);ns['remote_pair']=remote_single
    configure=types.FunctionType(batch.configure_app.__code__,ns)
    return configure(modal,path,plan,bound)


def execute(path,sha):
    ns=dict(batch.execute.__globals__)
    ns.update(plan_binding=plan_binding,configure_app=configure_app)
    return types.FunctionType(batch.execute.__code__,ns)(path,sha)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run",type=Path);g.add_argument("--owner",action="store_true")
    g.add_argument("--game-child",type=int,choices=(1,))
    p.add_argument("--sha");p.add_argument("--expires",type=float);p.add_argument("--image-id")
    a=p.parse_args()
    if a.owner:
        bound,plan=plan_binding(batch.base.PLAN,a.sha,remote=True)
        if not 0<batch.base.wire.remaining(a.expires)<=bound.OWNER_SECONDS+1: raise ValueError("bounded owner required")
        bound.owner(plan,a.sha,a.expires,a.image_id)
    elif a.game_child is not None:
        bound,_=plan_binding(batch.base.PLAN,batch.base.wire.digest(batch.base.PLAN),remote=True)
        bound.game_child(1)
    else:
        import json
        print(json.dumps(execute(a.run,a.sha)))
