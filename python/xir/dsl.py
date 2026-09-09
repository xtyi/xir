"""Restricted Python frontend for the agent-facing schedule AST."""
import ast
import inspect
import json
import textwrap
from dataclasses import dataclass


class DSLParseError(ValueError):
    code = "XIR001"

    def __init__(self, message, filename, node):
        self.filename = filename
        self.line = getattr(node, "lineno", 1)
        self.column = getattr(node, "col_offset", 0)
        super().__init__(f"{filename}:{self.line}:{self.column}: {message}")


class Node:
    def __init__(self, kind, attrs=None, children=None):
        self.kind, self.attrs, self.children = kind, attrs or {}, children or []

    def to_dict(self):
        result = {"kind": self.kind, **self.attrs}
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


class _Builder:
    kinds = {"shared": "BufferDecl", "pipeline": "PipelineDecl", "role": "RoleDecl", "async_copy": "AsyncCopy", "wait": "Wait", "sqmma": "Sqmma"}

    def __init__(self, filename):
        self.filename = filename

    def error(self, message, node):
        return DSLParseError(message, self.filename, node)

    def expr(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
            return node.value
        if isinstance(node, ast.Name):
            return {"symbol": node.id}
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self.expr(x) for x in node.elts]
        raise self.error("unsupported expression in restricted DSL", node)

    def call(self, call, binding=None):
        if not isinstance(call.func, ast.Name) or call.func.id not in self.kinds:
            name = getattr(call.func, "id", "<expression>")
            raise self.error(f"unknown DSL operation '{name}'", call.func)
        attrs = {"args": [self.expr(x) for x in call.args], "kwargs": {x.arg: self.expr(x.value) for x in call.keywords}}
        if binding:
            attrs["name"] = binding
        return Node(self.kinds[call.func.id], attrs)

    def stmt(self, node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
            return self.call(node.value, node.targets[0].id)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return self.call(node.value)
        if isinstance(node, ast.With) and len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call):
            context = node.items[0].context_expr
            if not isinstance(context.func, ast.Name) or context.func.id != "role":
                raise self.error("only with role(...) is supported", context)
            return Node("RoleRegion", {"role": self.expr(context.args[0]) if context.args else None}, [self.stmt(x) for x in node.body])
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "stages":
            return Node("StageFor", {"iterator": node.target.id, "pipeline": self.expr(node.iter.args[0]) if node.iter.args else None}, [self.stmt(x) for x in node.body])
        raise self.error("unsupported statement in restricted DSL", node)

    def build(self, fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)), filename=self.filename)
        definition = next(x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)))
        return Node("Kernel", {"name": fn.__name__, "params": [{"name": x.arg} for x in definition.args.args]}, [self.stmt(x) for x in definition.body])


@dataclass
class KernelProgram:
    function: object
    filename: str
    _ast: Node | None = None

    def build(self):
        self._ast = _Builder(self.filename).build(self.function)
        return self

    def to_dict(self):
        if self._ast is None:
            self.build()
        return self._ast.to_dict()

    def dump(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def kernel(fn):
    return KernelProgram(fn, inspect.getsourcefile(fn) or "<kernel>")
