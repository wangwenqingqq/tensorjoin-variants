"""CPU-only AST equivalence and transparent kernel-proxy checks."""

import ast
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent


class Normalize(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == 'g14_diagnostics' else node

    def visit_Expr(self, node):
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name) and value.func.value.id == 'diag'):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & {'diag', 'diagnostic', 'public_stopped'}:
            return None
        if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and t.value.id == 'diag' for t in node.targets):
            return None
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'KernelProxy'):
            return None
        if 'result_path' in names:
            node.value = ast.Constant('OUTPUT_PATH_RELOCATION')
        if 'public_seconds' in names:
            node.value = ast.parse('time.perf_counter() - public_started', mode='eval').body
        return self.generic_visit(node)

    def visit_If(self, node):
        if (isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                and node.test.left.id == 'visible'):
            return ast.parse('if visible != PHYSICAL_GPU: raise RuntimeError("GPU_SELECTOR_ONLY")').body[0]
        return self.generic_visit(node)

    def visit_Dict(self, node):
        kept = [(k, v) for k, v in zip(node.keys, node.values)
                if not (isinstance(k, ast.Constant) and k.value == 'g14_diagnostic')]
        node.keys, node.values = [k for k, v in kept], [v for k, v in kept]
        return self.generic_visit(node)


def main():
    for method in ('g2b', 'g5'):
        source = PROJECT / f'src/run_{method}_tensorjoin_public.py'
        adapter = HERE / f'src/run_{method}_diagnostic.py'
        original = Normalize().visit(ast.parse(source.read_text()))
        transformed = Normalize().visit(ast.parse(adapter.read_text()))
        assert ast.dump(original) == ast.dump(transformed), method
    sys.modules['torch'] = types.SimpleNamespace()
    from g14_diagnostics import KernelProxy
    calls = []

    class Event:
        def record(self):
            calls.append('event')

    class Kernel:
        def __getitem__(self, grid):
            def invoke(*args, **kwargs):
                calls.append((grid, args, kwargs))
                return 17
            return invoke

    diagnostic = types.SimpleNamespace(active=False)
    proxy = KernelProxy(Kernel(), 'stage1', diagnostic)
    sentinel = object()
    assert proxy[(3,)](sentinel, N=8) == 17
    assert calls == [((3,), (sentinel,), {'N': 8})]
    calls.clear()
    diagnostic.active = True
    diagnostic.indices = {'stage1': 0}
    diagnostic.events = {'stage1': [(Event(), Event())]}
    diagnostic.launched = []
    diagnostic.batch = 9
    assert proxy[(3,)](sentinel, N=8) == 17
    assert calls == ['event', ((3,), (sentinel,), {'N': 8}), 'event']
    assert diagnostic.launched == [('stage1', 0, 9)]
    print('PASS: both adapters preserve normalized semantics; proxies preserve dispatch arguments/count')


if __name__ == '__main__':
    main()
