"""Minimal Triton elementwise add example.

This file intentionally keeps the core API small:

- ``add_kernel`` is the Triton kernel.
- ``add`` launches it for existing CUDA buffers.
- running this file performs a no-extra-dependency smoke test through the CUDA
  Driver API.
"""

import ctypes
import ctypes.util
from array import array
from dataclasses import dataclass
from typing import Any

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    a_ptr: "tl.float32*",
    b_ptr: "tl.float32*",
    c_ptr: "tl.float32*",
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def add(a: Any, b: Any, c: Any, n_elements: int, *, block_size: int = 1024) -> None:
    """Launch ``c[i] = a[i] + b[i]`` for contiguous fp32 CUDA buffers.

    ``a``, ``b`` and ``c`` may be raw device pointer integers or objects with a
    ``data_ptr()`` method, such as torch tensors. The wrapper deliberately does
    not allocate memory or infer tensor length; callers own the ABI contract.
    """

    if not isinstance(n_elements, int) or isinstance(n_elements, bool):
        raise TypeError("n_elements must be an int")
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise TypeError("block_size must be an int")
    if n_elements < 0:
        raise ValueError("n_elements must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n_elements == 0:
        return

    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=block_size)


class CudaError(RuntimeError):
    pass


class _CudaDriver:
    CUDA_SUCCESS = 0

    def __init__(self) -> None:
        library_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        self.lib = ctypes.CDLL(library_name)
        self._bind()
        self._check(self.lib.cuInit(0), "cuInit")

        device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

        self.context = ctypes.c_void_p()
        self._check(
            self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), device),
            "cuDevicePrimaryCtxRetain",
        )
        self._check(self.lib.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")
        self._released = False

    def _bind(self) -> None:
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int

        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGet.restype = ctypes.c_int

        self.lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self.lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int

        self.lib.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        self.lib.cuDevicePrimaryCtxRelease.restype = ctypes.c_int

        self.lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.lib.cuCtxSetCurrent.restype = ctypes.c_int

        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuCtxSynchronize.restype = ctypes.c_int

        self.lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.lib.cuMemAlloc_v2.restype = ctypes.c_int

        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuMemFree_v2.restype = ctypes.c_int

        self.lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuMemcpyHtoD_v2.restype = ctypes.c_int

        self.lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.lib.cuMemcpyDtoH_v2.restype = ctypes.c_int

        self.lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuGetErrorString.restype = ctypes.c_int

    def allocate(self, byte_count: int) -> "DeviceBuffer":
        return DeviceBuffer(self, byte_count)

    def synchronize(self) -> None:
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self) -> None:
        if not self._released:
            self._check(self.lib.cuDevicePrimaryCtxRelease(0), "cuDevicePrimaryCtxRelease")
            self._released = True

    def _check(self, status: int, operation: str) -> None:
        if status == self.CUDA_SUCCESS:
            return
        message = ctypes.c_char_p()
        if (
            self.lib.cuGetErrorString(status, ctypes.byref(message)) == self.CUDA_SUCCESS
            and message.value
        ):
            detail = message.value.decode("utf-8", errors="replace")
        else:
            detail = f"CUDA error {status}"
        raise CudaError(f"{operation} failed: {detail}")


@dataclass
class DeviceBuffer:
    driver: _CudaDriver
    byte_count: int
    dtype = "float32"

    def __post_init__(self) -> None:
        self.ptr = ctypes.c_uint64()
        self.driver._check(
            self.driver.lib.cuMemAlloc_v2(ctypes.byref(self.ptr), self.byte_count),
            "cuMemAlloc_v2",
        )
        self._freed = False

    def data_ptr(self) -> int:
        return int(self.ptr.value)

    def copy_from_f32(self, values: array) -> None:
        if values.typecode != "f":
            raise TypeError("copy_from_f32 expects array('f')")
        byte_count = len(values) * values.itemsize
        if byte_count > self.byte_count:
            raise ValueError("source array is larger than device buffer")
        self.driver._check(
            self.driver.lib.cuMemcpyHtoD_v2(self.data_ptr(), values.buffer_info()[0], byte_count),
            "cuMemcpyHtoD_v2",
        )

    def copy_to_f32(self, count: int) -> array:
        result = array("f", [0.0]) * count
        byte_count = len(result) * result.itemsize
        if byte_count > self.byte_count:
            raise ValueError("destination array is larger than device buffer")
        self.driver._check(
            self.driver.lib.cuMemcpyDtoH_v2(result.buffer_info()[0], self.data_ptr(), byte_count),
            "cuMemcpyDtoH_v2",
        )
        return result

    def close(self) -> None:
        if not self._freed:
            self.driver._check(self.driver.lib.cuMemFree_v2(self.data_ptr()), "cuMemFree_v2")
            self._freed = True

    def __enter__(self) -> "DeviceBuffer":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _host_values(n_elements: int) -> tuple[array, array]:
    a = array("f", (float(i) * 0.5 for i in range(n_elements)))
    b = array("f", (float(i % 7) - 3.0 for i in range(n_elements)))
    return a, b


def _run_smoke_test() -> None:
    sizes = [0, 1, 31, 32, 33, 1024, 1025]
    driver = _CudaDriver()
    try:
        for n_elements in sizes:
            if n_elements == 0:
                add(0, 0, 0, 0, block_size=256)
                continue

            a_host, b_host = _host_values(n_elements)
            byte_count = n_elements * a_host.itemsize
            with driver.allocate(byte_count) as a_dev:
                with driver.allocate(byte_count) as b_dev:
                    with driver.allocate(byte_count) as c_dev:
                        a_dev.copy_from_f32(a_host)
                        b_dev.copy_from_f32(b_host)
                        if n_elements % 2 == 0:
                            add(
                                a_dev.data_ptr(),
                                b_dev.data_ptr(),
                                c_dev.data_ptr(),
                                n_elements,
                                block_size=256,
                            )
                        else:
                            add(a_dev, b_dev, c_dev, n_elements, block_size=256)
                        driver.synchronize()
                        c_host = c_dev.copy_to_f32(n_elements)

            expected = array("f", (x + y for x, y in zip(a_host, b_host)))
            if c_host != expected:
                raise AssertionError(f"mismatch for n_elements={n_elements}")
    finally:
        driver.close()

    print(f"validated Triton elementwise add for sizes: {sizes}")


if __name__ == "__main__":
    _run_smoke_test()
