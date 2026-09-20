"""Minimal nvcc + ctypes transport for the admitted CUDA elementwise subset."""
from array import array
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from .dsl import IRValidationError, _native


class CudaCompileError(RuntimeError):
    pass


class CudaRuntimeError(RuntimeError):
    pass


def compile_cuda(program, output_dir, *, nvcc="nvcc", verifier=None):
    """Validate a snapshot, emit CUDA, compile a shared library, and retain evidence.

    Compilation does not run a kernel. There is no performance-oriented cache,
    device-pointer ABI, asynchronous execution, or implicit dtype conversion.
    """
    module = program.to_dict(verifier=verifier)
    response = _native("emit-cuda", module, verifier)
    if not response["ok"]:
        raise IRValidationError(response["diagnostics"])
    artifact = response["artifact"]
    compiler = shutil.which(str(nvcc))
    if compiler is None:
        raise CudaCompileError(f"CUDA compiler not found: {nvcc}")
    version_result = subprocess.run([compiler, "--version"], text=True, capture_output=True, check=False)
    if version_result.returncode:
        raise CudaCompileError(f"Cannot query nvcc: {version_result.stderr}")
    flags = ["-std=c++17", "-O2", "--fmad=false", "-lineinfo", f"-arch={artifact['arch']}", "--shared", "-Xcompiler=-fPIC", "--cudart=static"]
    identity = json.dumps({"module": module, "source": artifact["source"], "compiler": compiler, "version": version_result.stdout, "flags": flags}, sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    directory = Path(output_dir).resolve() / digest[:16]
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "kernel.cu"
    library_path = directory / "kernel.so"
    source_path.write_text(artifact["source"])
    (directory / "module.json").write_text(json.dumps(module, indent=2))
    (directory / "artifact.json").write_text(json.dumps(artifact, indent=2))
    # Atomic replacement keeps existing ctypes mappings attached to their inode.
    with tempfile.TemporaryDirectory(prefix="compile-", dir=directory) as temporary:
        temporary_library = Path(temporary) / "kernel.so"
        command = [compiler, *flags, str(source_path), "-o", str(temporary_library)]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        evidence = {"command": command, "nvcc_version": version_result.stdout, "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr, "identity_sha256": digest}
        (directory / "compile.json").write_text(json.dumps(evidence, indent=2))
        if result.returncode:
            raise CudaCompileError(f"nvcc failed; details: {directory / 'compile.json'}\n{result.stderr}")
        temporary_library.replace(library_path)
    return CompiledCudaKernel(library_path, artifact)


class CompiledCudaKernel:
    def __init__(self, library_path, artifact):
        self.library_path = Path(library_path)
        self.source_path = self.library_path.with_name("kernel.cu")
        self._artifact = copy.deepcopy(artifact)
        try:
            self._library = ctypes.CDLL(str(self.library_path))
        except OSError as error:
            raise CudaRuntimeError(f"Cannot load compiled CUDA library: {error}") from error
        self._launch = getattr(self._library, artifact["entrypoint"])
        self._launch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint64),
                                ctypes.c_uint64, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
        self._launch.restype = ctypes.c_int

    def run(self, *arguments):
        """Synchronous host-array execution in declared parameter order.

        Pointer arguments must be distinct array('f') objects. Mutable pointer
        arguments are copied back in place; const pointer arguments are inputs.
        Transfers and allocation are deliberately visible costs of this API.
        """
        params = self._artifact["params"]
        if len(arguments) != len(params):
            raise ValueError(f"expected {len(params)} arguments, got {len(arguments)}")
        n = arguments[self._artifact["extent_parameter"]]
        if type(n) is not int or not 0 <= n <= (2**64 - 1) // 4:
            raise ValueError("extent must be a nonnegative integer whose FP32 byte size fits uint64")
        pointers = (ctypes.c_void_p * len(params))()
        lengths = (ctypes.c_uint64 * len(params))()
        seen = set()
        # Keep arguments alive until the synchronous native call has completed.
        for position, (param, argument) in enumerate(zip(params, arguments)):
            if param["ty"] == "Index":
                continue
            if not isinstance(argument, array) or argument.typecode != "f" or argument.itemsize != 4:
                raise TypeError(f"parameter {param['id']} requires array('f') with 4-byte elements")
            if len(argument) < n:
                raise ValueError(f"parameter {param['id']} has {len(argument)} elements, requires at least {n}")
            if id(argument) in seen:
                raise ValueError("pointer arguments must be distinct host arrays in the initial CUDA ABI")
            seen.add(id(argument))
            address, length = argument.buffer_info()
            pointers[position], lengths[position] = address, length
        error = ctypes.create_string_buffer(4096)
        status = self._launch(pointers, lengths, n, error, len(error))
        if status:
            raise CudaRuntimeError(error.value.decode(errors="replace"))
