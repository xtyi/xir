"""Restricted source frontend for schema v2; Rust owns IR validation.

Source is parsed, never evaluated. Resource handles, events and tickets have
separate kinds, while rebinding an ordinary value creates a fresh SSA value.
"""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap


class DSLParseError(ValueError):
    code = "UnsupportedSyntax"

    def __init__(self, message, filename, node):
        self.filename = filename
        self.line = getattr(node, "lineno", 1)
        self.column = getattr(node, "col_offset", 0)
        super().__init__(f"{filename}:{self.line}:{self.column}: {message}")


class IRValidationError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("\n".join(f"{item['code']}: {item['message']}" for item in diagnostics))


class NativeUnavailableError(RuntimeError):
    pass


def _native(command, module, executable=None):
    requested = executable or os.environ.get("XIR_VERIFY_BIN")
    if requested:
        binary = str(requested)
    else:
        local = Path(__file__).resolve().parents[2] / "target" / "debug" / "xir-verify"
        binary = shutil.which("xir-verify") or (str(local) if local.is_file() else None)
    if binary is None:
        raise NativeUnavailableError("Rust verifier is required. Run `cargo build --bin xir-verify` and set XIR_VERIFY_BIN if needed.")
    try:
        result = subprocess.run([binary, command], input=json.dumps(module, allow_nan=False), text=True, capture_output=True, check=False)
    except OSError as error:
        raise NativeUnavailableError(f"Cannot execute Rust verifier {binary}: {error}") from error
    if result.returncode:
        raise RuntimeError(f"Rust verifier failed ({result.returncode}): {result.stderr.strip()}")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Rust verifier returned invalid JSON") from error
    if "ok" not in response:
        raise RuntimeError("Rust verifier returned an incompatible response")
    return response


def _tag(kind, data):
    return {"kind": kind, "data": data}


_DTYPES = {name.lower(): name for name in ("Bool", "I32", "I64", "U32", "U64", "F16", "BF16", "F32", "F64")}
_TYPES = {name: {"Scalar": dtype} for name, dtype in _DTYPES.items()}
_TYPES.update(index="Index", predicate="Predicate", bool="Predicate")


class _Symbol:
    def __init__(self, kind, identity, ty=None):
        self.kind, self.id, self.ty = kind, identity, ty

    def definition(self):
        return {"id": self.id, "ty": self.ty}


class _Builder:
    def __init__(self, filename, block):
        self.filename = filename
        self.block = list(block)
        self.counter = 0
        self.env = {}
        self.body = []
        self.context = "kernel"
        self.declarations_open = True
        self.declarations = {name: [] for name in ("storages", "buffers", "roles", "pipelines", "accumulators")}

    def error(self, message, node):
        return DSLParseError(message, self.filename, node)

    def fresh(self, name, kind="value", ty=None):
        identity = f"{name}.{self.counter}"
        self.counter += 1
        return _Symbol(kind, identity, ty)

    def emit(self, kind, data, source):
        self.body.append({"meta": {"node_id": "", "node_revision": 0, "origin_ids": [], "source_span": {
            "file": self.filename, "start_line": source.lineno, "start_column": source.col_offset,
            "end_line": getattr(source, "end_lineno", source.lineno), "end_column": getattr(source, "end_col_offset", source.col_offset),
        }}, "kind": {"kind": kind, **({"data": data} if data is not None else {})}})

    def symbol(self, node, kind="value"):
        if not isinstance(node, ast.Name) or node.id not in self.env:
            raise self.error("expected a defined symbol", node)
        symbol = self.env[node.id]
        if symbol.kind != kind:
            raise self.error(f"expected {kind}, got {symbol.kind}", node)
        return symbol

    def literal(self, node):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError) as error:
            raise self.error("expected a compile-time literal", node) from error

    def type_annotation(self, node):
        name = node.id if isinstance(node, ast.Name) else self.literal(node)
        if not isinstance(name, str):
            raise self.error("type annotation must be a supported type name", node)
        if name in _TYPES:
            return copy.deepcopy(_TYPES[name])
        match = re.fullmatch(r"(const_ptr|ptr)\[([a-z0-9]+)\]", name)
        if match and match[2] in _DTYPES:
            return {"Pointer": {"element": _DTYPES[match[2]], "address_space": "Global", "mutable": match[1] == "ptr"}}
        raise self.error(f"unsupported type annotation {name!r}", node)

    def arguments(self, call, names, optional=()):
        if len(call.args) > len(names) or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise self.error("wrong argument count or argument unpacking", call)
        args = dict(zip(names, call.args))
        for keyword in call.keywords:
            if keyword.arg not in names or keyword.arg in args:
                raise self.error("unknown, duplicate or unpacked keyword argument", keyword.value)
            args[keyword.arg] = keyword.value
        if any(name not in args and name not in optional for name in names):
            raise self.error(f"required arguments: {', '.join(names)}", call)
        return args

    def index_expr(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0:
            return _tag("Constant", node.value)
        if isinstance(node, ast.Name):
            value = self.symbol(node)
            if value.ty != "Index":
                raise self.error("resource dimensions require index values", node)
            return _tag("Value", value.id)
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Mult, ast.FloorDiv):
            kind = {ast.Add: "Add", ast.Mult: "Mul", ast.FloorDiv: "FloorDiv"}[type(node.op)]
            return _tag(kind, [self.index_expr(node.left), self.index_expr(node.right)])
        raise self.error("expected nonnegative index expression", node)

    def sequence(self, node):
        if not isinstance(node, (ast.List, ast.Tuple)):
            raise self.error("expected an explicit list or tuple", node)
        return node.elts

    def memory(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "leased":
            args = self.arguments(node, ("buffer", "ticket"))
            buffer = self.symbol(args["buffer"], "buffer")
            ticket = self.symbol(args["ticket"], "ticket")
            return _tag("Leased", {"buffer": buffer.id, "ticket": ticket.id}), buffer.ty
        buffer = self.symbol(node, "buffer")
        return _tag("Buffer", buffer.id), buffer.ty

    def value(self, node, expected=None):
        if isinstance(node, ast.Name):
            return self.symbol(node)
        if isinstance(node, ast.Constant) or isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            value = self.literal(node)
            ty = expected or ("Predicate" if type(value) is bool else {"Scalar": "I32"} if type(value) is int else {"Scalar": "F32"})
            if type(value) is bool:
                kind = "Bool"
            elif type(value) is int:
                kind = "Index" if ty == "Index" else "Integer"
            elif type(value) is float:
                kind = "Float"
            else:
                raise self.error("only numeric and boolean values are supported", node)
            result = self.fresh("literal", ty=ty)
            self.emit("Constant", {"value": _tag(kind, value), "result": result.definition()}, node)
            return result
        if isinstance(node, ast.Call):
            result = self.operation(node)
            if result is None or result.kind != "value":
                raise self.error("operation does not return an ordinary value", node)
            return result
        if isinstance(node, (ast.BinOp, ast.Compare)):
            if isinstance(node, ast.Compare):
                if len(node.ops) != 1 or type(node.ops[0]) not in (ast.Lt, ast.Eq):
                    raise self.error("only single < and == comparisons are supported", node)
                left, right, op = node.left, node.comparators[0], {ast.Lt: "Less", ast.Eq: "Equal"}[type(node.ops[0])]
            else:
                ops = {ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mul", ast.Div: "Div", ast.FloorDiv: "Div"}
                if type(node.op) not in ops:
                    raise self.error("unsupported binary operator", node)
                left, right, op = node.left, node.right, ops[type(node.op)]
            hint = expected
            if isinstance(right, ast.Name) and right.id in self.env:
                hint = self.symbol(right).ty
            lhs = self.value(left, hint)
            rhs = self.value(right, lhs.ty)
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Div) and lhs.ty not in ({"Scalar": "F16"}, {"Scalar": "BF16"}, {"Scalar": "F32"}, {"Scalar": "F64"}):
                    raise self.error("/ requires floating values; use // for index division", node)
                if isinstance(node.op, ast.FloorDiv) and lhs.ty != "Index":
                    raise self.error("// currently requires unsigned index values", node)
            result = self.fresh("binary", ty="Predicate" if op in ("Less", "Equal") else lhs.ty)
            self.emit("Binary", {"op": op, "lhs": lhs.id, "rhs": rhs.id, "result": result.definition()}, node)
            return result
        raise self.error("unsupported value expression", node)

    def operation(self, call):
        if not isinstance(call.func, ast.Name):
            raise self.error("DSL operations must be simple names", call)
        name = call.func.id
        if name in _TYPES:
            args = self.arguments(call, ("value",))
            if not isinstance(args["value"], (ast.Constant, ast.UnaryOp)):
                raise self.error("typed literal constructors do not perform casts", call)
            return self.value(args["value"], copy.deepcopy(_TYPES[name]))
        declarations = {"storage", "buffer", "role", "pipeline", "accumulator"}
        if name in declarations:
            if self.context != "kernel" or not self.declarations_open:
                raise self.error("resource declarations must precede executable statements at kernel scope", call)
            return self.declaration(name, call)
        self.declarations_open = False
        if name == "load":
            args = self.arguments(call, ("source", "indices", "mask", "masked_value"))
            source, buffer = self.memory(args["source"])
            ty = {"Scalar": buffer["element"]}
            result = self.fresh("load", ty=ty)
            self.emit("Load", {"source": source, "indices": [self.value(n, "Index").id for n in self.sequence(args["indices"])],
                "mask": self.value(args["mask"], "Predicate").id, "masked_value": self.value(args["masked_value"], ty).id, "result": result.definition()}, call)
            return result
        if name == "store":
            args = self.arguments(call, ("destination", "indices", "value", "mask"))
            destination, buffer = self.memory(args["destination"])
            self.emit("Store", {"destination": destination, "indices": [self.value(n, "Index").id for n in self.sequence(args["indices"])],
                "value": self.value(args["value"], {"Scalar": buffer["element"]}).id, "mask": self.value(args["mask"], "Predicate").id}, call)
            return None
        if name in ("acquire", "consume"):
            args = self.arguments(call, ("pipeline", "iteration"))
            result = self.fresh(name, "ticket")
            self.emit("PipelineAcquire" if name == "acquire" else "PipelineConsume", {"pipeline": self.symbol(args["pipeline"], "pipeline").id,
                "iteration": self.value(args["iteration"], "Index").id, "ticket": result.id}, call)
            return result
        if name in ("publish", "release", "drain", "await_event"):
            names = {"publish": ("ticket", "events"), "release": ("ticket",), "drain": ("pipeline",), "await_event": ("event",)}[name]
            args = self.arguments(call, names)
            data = {names[0]: self.symbol(args[names[0]], names[0]).id}
            if name == "publish":
                data["events"] = [self.symbol(event, "event").id for event in self.sequence(args["events"])]
            self.emit({"publish": "PipelinePublish", "release": "PipelineRelease", "drain": "PipelineDrain", "await_event": "AwaitEvent"}[name], data, call)
            return None
        if name == "async_copy":
            args = self.arguments(call, ("source", "destination", "transfer"))
            result = self.fresh("copy", "event")
            self.emit("AsyncCopy", {"source": self.memory(args["source"])[0], "destination": self.memory(args["destination"])[0],
                "transfer": self.literal(args["transfer"]), "completion": result.id}, call)
            return result
        if name in ("accumulator_init", "accumulator_read", "sqmma"):
            names = {"accumulator_init": ("accumulator", "value"), "accumulator_read": ("accumulator",), "sqmma": ("accumulator", "a", "b", "instruction")}[name]
            args = self.arguments(call, names)
            accumulator = self.symbol(args["accumulator"], "accumulator")
            data = {"accumulator": accumulator.id}
            if name == "accumulator_init":
                data["value"] = self.value(args["value"], {"Scalar": accumulator.ty["element"]}).id
                self.emit("AccumulatorInit", data, call)
                return None
            if name == "accumulator_read":
                result = self.fresh("fragment", ty={"Fragment": accumulator.ty})
                data["result"] = result.definition()
                self.emit("AccumulatorRead", data, call)
                return result
            result = self.fresh("mma", "event")
            data.update(a=self.memory(args["a"])[0], b=self.memory(args["b"])[0], instruction=self.literal(args["instruction"]), completion=result.id)
            self.emit("Sqmma", data, call)
            return result
        raise self.error(f"unknown DSL operation '{name}'; legacy stages/wait syntax is not supported by schema v2", call)

    def declaration(self, name, call):
        signatures = {
            "storage": ("address_space", "capacity_bytes", "alignment_bytes", "external"),
            "buffer": ("storage", "dtype", "shape", "strides_bytes", "offset_bytes", "mutable"),
            "role": ("warps",),
            "pipeline": ("iterations", "slots", "producer", "consumer", "buffers", "slot_strides_bytes"),
            "accumulator": ("dtype", "shape", "participants", "representation"),
        }
        args = self.arguments(call, signatures[name], optional=("external",) if name == "storage" else ())
        result = self.fresh(name, name)
        data = {"id": result.id}
        if name == "storage":
            ty = {"address_space": self.literal(args["address_space"]), "capacity_bytes": self.index_expr(args["capacity_bytes"]), "alignment_bytes": self.literal(args["alignment_bytes"])}
            data.update(ty=ty, external=self.symbol(args["external"]).id if "external" in args else None)
        elif name in ("buffer", "accumulator"):
            dtype = self.literal(args["dtype"])
            if not isinstance(dtype, str) or dtype not in _DTYPES:
                raise self.error("unknown dtype", args["dtype"])
            shape = [self.index_expr(n) for n in self.sequence(args["shape"])]
            ty = {"element": _DTYPES[dtype]}
            if name == "buffer":
                ty.update(storage=self.symbol(args["storage"], "storage").id, offset_bytes=self.index_expr(args["offset_bytes"]), shape=shape,
                    mapping=_tag("Strided", {"strides_bytes": [self.index_expr(n) for n in self.sequence(args["strides_bytes"])]}), mutable=self.literal(args["mutable"]))
            else:
                ty.update(logical_shape=shape, participants=self.symbol(args["participants"], "role").id, representation=self.literal(args["representation"]))
            data["ty"] = ty
        elif name == "role":
            ty = None
            data["warps"] = self.literal(args["warps"])
        else:
            buffers = self.sequence(args["buffers"])
            strides = self.sequence(args["slot_strides_bytes"])
            if len(buffers) != len(strides):
                raise self.error("each pipeline buffer requires one slot stride", call)
            ty = None
            data.update(iteration_count=self.index_expr(args["iterations"]), slots=self.literal(args["slots"]),
                producer=self.symbol(args["producer"], "role").id, consumer=self.symbol(args["consumer"], "role").id,
                buffers=[{"buffer": self.symbol(buffer, "buffer").id, "slot_stride_bytes": self.literal(stride)} for buffer, stride in zip(buffers, strides)])
        result.ty = ty
        self.declarations[{"storage": "storages", "buffer": "buffers", "role": "roles", "pipeline": "pipelines", "accumulator": "accumulators"}[name]].append(data)
        return result

    def bind(self, name, symbol, source):
        if symbol is None:
            raise self.error("operation has no result to bind", source)
        old = self.env.get(name)
        if old is not None and (old.kind != "value" or symbol.kind != "value"):
            raise self.error("resource, event and ticket bindings cannot be reassigned", source)
        self.env[name] = symbol

    def nested(self, statements, env, context, ending=None):
        outer_body, outer_env, outer_context = self.body, self.env, self.context
        self.body, self.env, self.context = [], env.copy(), context
        for statement in statements:
            self.statement(statement)
        if ending is not None:
            values = [self.env[name].id for name in ending]
            self.emit("Yield", {"values": values}, statements[-1])
        result = self.body, self.env
        self.body, self.env, self.context = outer_body, outer_env, outer_context
        return result

    def assigned_names(self, statements):
        # Loop induction variables have lexical scope and are not loop-carried.
        names = set()
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                names.update(target.id for target in targets if isinstance(target, ast.Name))
            elif isinstance(statement, (ast.If, ast.For)):
                names.update(self.assigned_names(statement.body))
                names.update(self.assigned_names(statement.orelse))
        return names

    def statement(self, statement):
        if isinstance(statement, ast.Pass):
            return
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name) or statement.value is None:
                raise self.error("assignment requires one name and a value", statement)
            expected = self.type_annotation(statement.annotation) if isinstance(statement, ast.AnnAssign) else None
            declaration = isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id in {"storage", "buffer", "role", "pipeline", "accumulator"}
            if not declaration:
                self.declarations_open = False
            if isinstance(statement.value, ast.Call):
                result = self.operation(statement.value)
            else:
                result = self.value(statement.value, expected)
            if expected is not None and (result is None or result.kind != "value" or result.ty != expected):
                raise self.error("annotated assignment type does not match its value", statement)
            self.bind(targets[0].id, result, statement)
            return
        self.declarations_open = False
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            if self.operation(statement.value) is not None:
                raise self.error("operation result must be bound explicitly", statement)
            return
        if isinstance(statement, ast.If):
            condition = self.value(statement.test, "Predicate")
            assigned = self.assigned_names(statement.body + statement.orelse)
            then_body, then_env = self.nested(statement.body, self.env, "control")
            else_body, else_env = self.nested(statement.orelse, self.env, "control")
            merged = sorted(name for name in assigned if name in then_env and name in else_env)
            results = []
            for name in merged:
                left, right = then_env[name], else_env[name]
                if left.kind != "value" or right.kind != "value" or left.ty != right.ty:
                    raise self.error("If joins require ordinary values of matching types", statement)
                result = self.fresh(name, ty=left.ty)
                results.append(result.definition())
                self.env[name] = result
            outer = self.body
            for body, env in ((then_body, then_env), (else_body, else_env)):
                self.body = body
                self.emit("Yield", {"values": [env[name].id for name in merged]}, statement)
            self.body = outer
            self.emit("If", {"condition": condition.id, "results": results, "then_body": then_body, "else_body": else_body}, statement)
            return
        if isinstance(statement, ast.For):
            if statement.orelse or not isinstance(statement.target, ast.Name) or not isinstance(statement.iter, ast.Call) or not isinstance(statement.iter.func, ast.Name):
                raise self.error("For requires a name, range/pipeline_iterations, and no else clause", statement)
            iterator = statement.iter
            induction = self.fresh(statement.target.id, ty="Index")
            env = self.env.copy()
            if statement.target.id in env and env[statement.target.id].kind != "value":
                raise self.error("induction cannot shadow a resource handle", statement.target)
            env[statement.target.id] = induction
            if iterator.func.id == "pipeline_iterations":
                args = self.arguments(iterator, ("pipeline",))
                pipeline = self.symbol(args["pipeline"], "pipeline")
                body, _ = self.nested(statement.body, env, "pipeline")
                self.emit("PipelineFor", {"pipeline": pipeline.id, "iteration": induction.definition(), "body": body}, statement)
                return
            if iterator.func.id != "range" or iterator.keywords or len(iterator.args) not in (1, 2, 3):
                raise self.error("expected range with one to three positional bounds or pipeline_iterations(pipeline)", iterator)
            def constant(value):
                return ast.copy_location(ast.Constant(value=value), iterator)
            args = iterator.args
            lower, upper, step = (constant(0), args[0], constant(1)) if len(args) == 1 else (args[0], args[1], args[2] if len(args) == 3 else constant(1))
            lower, upper, step = [self.value(bound, "Index").id for bound in (lower, upper, step)]
            names = sorted((self.assigned_names(statement.body) & self.env.keys()) - {statement.target.id})
            carried = []
            for name in names:
                initial = self.env[name]
                if initial.kind != "value":
                    raise self.error("only ordinary values may be loop-carried", statement)
                argument = self.fresh(name + "_arg", ty=initial.ty)
                result = self.fresh(name + "_result", ty=initial.ty)
                carried.append({"initial": initial.id, "argument": argument.definition(), "result": result.definition()})
                env[name] = argument
            body, _ = self.nested(statement.body, env, "control", ending=names)
            self.emit("For", {"lower": lower, "upper": upper, "step": step, "induction": induction.definition(), "carried": carried, "body": body}, statement)
            for name, item in zip(names, carried):
                self.env[name] = _Symbol("value", item["result"]["id"], item["result"]["ty"])
            return
        if isinstance(statement, ast.With):
            call = self.context_call(statement, "concurrent_roles")
            self.arguments(call, ())
            if self.context != "kernel":
                raise self.error("concurrent_roles must be at kernel scope", statement)
            roles = []
            for region in statement.body:
                role_call = self.context_call(region, "role_region")
                args = self.arguments(role_call, ("role",))
                role = self.symbol(args["role"], "role")
                body, _ = self.nested(region.body, self.env, "role")
                roles.append({"role": role.id, "body": body})
            self.emit("ConcurrentRoles", {"roles": roles, "join_scope": "Block"}, statement)
            return
        raise self.error("unsupported statement in restricted DSL", statement)

    def context_call(self, statement, name):
        if not isinstance(statement, ast.With) or len(statement.items) != 1 or statement.items[0].optional_vars is not None:
            raise self.error(f"expected with {name}(...), without an as binding", statement)
        call = statement.items[0].context_expr
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != name:
            raise self.error(f"expected with {name}(...)", statement)
        return call

    def build(self, source, first_line=1):
        dedented = textwrap.dedent(source)
        original_lines = source.splitlines()
        parsed_lines = dedented.splitlines()
        removed_columns = [len(original.encode("utf-8")) - len(parsed.encode("utf-8"))
                           for original, parsed in zip(original_lines, parsed_lines)]
        try:
            tree = ast.parse(dedented, filename=self.filename)
        except SyntaxError as error:
            error.lineno = (error.lineno or 1) + first_line - 1
            raise error
        for node in ast.walk(tree):
            if hasattr(node, "lineno"):
                node.col_offset += removed_columns[node.lineno - 1]
            if getattr(node, "end_lineno", None) is not None:
                node.end_col_offset += removed_columns[node.end_lineno - 1]
        ast.increment_lineno(tree, first_line - 1)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise self.error("source must contain exactly one synchronous kernel function", tree.body[0] if tree.body else tree)
        fn = tree.body[0]
        for decorator in fn.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "kernel":
                continue
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "kernel":
                args = self.arguments(decorator, ("block",), optional=("block",))
                if "block" in args:
                    declared = list(self.literal(args["block"]))
                    if self.block and self.block != declared:
                        raise self.error("source decorator and requested block dimensions differ", decorator)
                    self.block = declared
                continue
            raise self.error("only the kernel decorator is supported", decorator)
        if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults or fn.args.kw_defaults:
            raise self.error("kernel parameters cannot have defaults, variadics or keyword-only arguments", fn)
        if fn.returns is not None and not (isinstance(fn.returns, ast.Constant) and fn.returns.value is None):
            raise self.error("kernel return annotation must be None", fn.returns)
        params = []
        for arg in fn.args.posonlyargs + fn.args.args:
            if arg.annotation is None:
                raise self.error("kernel parameters require explicit type annotations", arg)
            symbol = self.fresh(arg.arg, ty=self.type_annotation(arg.annotation))
            if arg.arg in self.env:
                raise self.error("duplicate parameter", arg)
            self.env[arg.arg] = symbol
            params.append(symbol.definition())
        for position, statement in enumerate(fn.body):
            if isinstance(statement, ast.Return):
                if position != len(fn.body) - 1 or statement.value is not None:
                    raise self.error("only a final bare return is supported", statement)
            else:
                self.statement(statement)
        self.emit("KernelReturn", None, fn.body[-1])
        return {"name": fn.name, "schema_version": 2, "module_revision": 0,
            "target_request": {"backend": "musa", "arch": "mp31"},
            "kernels": [{"name": fn.name, "params": params, "block": self.block or [1, 1, 1], **self.declarations, "body": self.body}]}


class KernelProgram:
    def __init__(self, source=None, filename="<kernel>", first_line=1, block=None, module=None):
        self.source, self.filename, self.first_line = source, filename, first_line
        self.block = block
        self._module = copy.deepcopy(module)

    @classmethod
    def from_dict(cls, module):
        """Import canonical JSON data; build/verify still go through Rust."""
        return cls(module=module)

    def build(self, *, verifier=None):
        if self._module is None:
            self._module = _Builder(self.filename, self.block or []).build(self.source, self.first_line)
        response = _native("construct", self._module, verifier)
        if not response["ok"]:
            raise IRValidationError(response["diagnostics"])
        self._module = response["module"]
        return self

    def to_dict(self, *, verifier=None):
        self.build(verifier=verifier)
        return copy.deepcopy(self._module)

    def dump(self, *, verifier=None):
        return json.dumps(self.to_dict(verifier=verifier), indent=2)

    def verify(self, *, verifier=None):
        self.build(verifier=verifier)
        return _native("verify", self._module, verifier)["report"]


def kernel_from_source(source, *, filename="<kernel>", block=None):
    """Source fallback for notebooks and generated code, without exec/eval."""
    return KernelProgram(source=source, filename=filename, block=block)


def kernel(fn=None, *, block=None):
    def decorate(function):
        try:
            lines, first_line = inspect.getsourcelines(function)
        except (OSError, TypeError) as error:
            raise ValueError("Kernel source is unavailable; use kernel_from_source(source).") from error
        return KernelProgram("".join(lines), inspect.getsourcefile(function) or "<kernel>", first_line, block)
    return decorate(fn) if fn is not None else decorate
