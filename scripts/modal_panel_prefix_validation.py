"""Reuse every proven per-game gate to validate a successful first game."""
import ast
import inspect
import textwrap


def validator(bound):
    tree = ast.parse(textwrap.dedent(inspect.getsource(bound.validate_terminal)))
    function = tree.body[0]
    early = [n for n in function.body if isinstance(n, ast.If)
             and ast.unparse(n.test) == "terminal['status'] == 'failed'"
             and len(n.body) == 1 and isinstance(n.body[0], ast.Return)]
    loops = [n for n in function.body if isinstance(n, ast.For)
             and ast.unparse(n.iter) == "enumerate(result['games'])"]
    if len(early) != 1 or len(loops) != 1:
        raise ValueError("proven validator structure changed")
    # The full two-child receipt remains intact. Check all setup gates and the
    # first child's complete scientific/cleanup gates. The caller separately
    # verifies that child1 failed strictly before gameplay.
    function.body.remove(early[0])
    loops[0].iter = ast.parse("enumerate(result['games'][:1])", mode="eval").body
    namespace = dict(bound.validate_terminal.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<validated-successful-prefix>", "exec"), namespace)
    return namespace["validate_terminal"]
