"""Typed owned-handle counterpart of the immutable H2B/G16 GEMM binding."""
import ctypes
import ctypes.util
import os
from pathlib import Path
import torch
from run_h2b_p1_pedantic_correctness import (sha256_file, CUBLAS_STATUS_SUCCESS,
    CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_POINTER_MODE_HOST, CUBLAS_PEDANTIC_MATH,
    CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT)

class OwnedCuBLAS:
    """Minimal typed binding for the frozen pedantic FP32 GEMM call."""

    def __init__(self) -> None:
        candidates = [
            str(
                Path(torch.__file__).resolve().parents[1]
                / "nvidia/cu13/lib/libcublas.so.13"
            ),
            "/usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13",
        ]
        found = ctypes.util.find_library("cublas")
        if found:
            candidates.append(found)
        error = None
        for candidate in candidates:
            try:
                self.library = ctypes.CDLL(candidate)
                candidate_path = Path(candidate)
                if not candidate_path.is_file():
                    raise OSError(f"loaded cuBLAS path is not hashable: {candidate}")
                self.library_path = candidate_path.resolve()
                break
            except OSError as caught:
                error = caught
        else:
            raise RuntimeError(f"unable to load libcublas: {error}")

        handle_type = ctypes.c_void_p
        self.library.cublasGetVersion_v2.argtypes = [handle_type, ctypes.POINTER(ctypes.c_int)]
        self.library.cublasGetVersion_v2.restype = ctypes.c_int
        self.library.cublasSetMathMode.argtypes = [handle_type, ctypes.c_int]
        self.library.cublasSetMathMode.restype = ctypes.c_int
        self.library.cublasGetMathMode.argtypes = [handle_type, ctypes.POINTER(ctypes.c_int)]
        self.library.cublasGetMathMode.restype = ctypes.c_int
        self.library.cublasSetPointerMode_v2.argtypes = [handle_type, ctypes.c_int]
        self.library.cublasSetPointerMode_v2.restype = ctypes.c_int
        self.library.cublasGetPointerMode_v2.argtypes = [
            handle_type,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.cublasGetPointerMode_v2.restype = ctypes.c_int
        self.library.cublasGemmEx.argtypes = [
            handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.cublasGemmEx.restype = ctypes.c_int

        signatures = {
            'cublasCreate_v2': [ctypes.POINTER(handle_type)],
            'cublasDestroy_v2': [handle_type],
            'cublasSetStream_v2': [handle_type, ctypes.c_void_p],
            'cublasGetStream_v2': [handle_type, ctypes.POINTER(ctypes.c_void_p)],
            'cublasSetWorkspace_v2': [handle_type, ctypes.c_void_p, ctypes.c_size_t],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments; function.restype = ctypes.c_int
        torch_library = ctypes.CDLL(str(Path(torch.__file__).resolve().parent / 'lib/libtorch_cuda.so'))
        query = getattr(torch_library, '_ZN2at4cuda22getChosenWorkspaceSizeEv')
        query.argtypes = []; query.restype = ctypes.c_size_t
        self.workspace_bytes = int(query())
        assert self.workspace_bytes == 8519680, self.workspace_bytes
        self.workspace = torch.empty(self.workspace_bytes, dtype=torch.uint8, device='cuda')
        self.stream = int(torch.cuda.current_stream().cuda_stream)
        self.handle = handle_type()
        self._check(self.library.cublasCreate_v2(ctypes.byref(self.handle)), 'cublasCreate_v2')
        self.handle_value = int(self.handle.value)
        self.closed = False
        self._check(self.library.cublasSetStream_v2(self.handle, ctypes.c_void_p(self.stream)), 'cublasSetStream_v2')
        self._check(self.library.cublasSetWorkspace_v2(self.handle, ctypes.c_void_p(self.workspace.data_ptr()), self.workspace_bytes), 'cublasSetWorkspace_v2')
        selected_stream = ctypes.c_void_p()
        self._check(self.library.cublasGetStream_v2(self.handle, ctypes.byref(selected_stream)), 'cublasGetStream_v2')
        assert int(selected_stream.value or 0) == self.stream
        self._check(
            self.library.cublasSetMathMode(self.handle, CUBLAS_PEDANTIC_MATH),
            "cublasSetMathMode",
        )
        self._check(
            self.library.cublasSetPointerMode_v2(
                self.handle, CUBLAS_POINTER_MODE_HOST
            ),
            "cublasSetPointerMode_v2",
        )
        version = ctypes.c_int()
        math_mode = ctypes.c_int()
        pointer_mode = ctypes.c_int()
        self._check(
            self.library.cublasGetVersion_v2(self.handle, ctypes.byref(version)),
            "cublasGetVersion_v2",
        )
        self._check(
            self.library.cublasGetMathMode(self.handle, ctypes.byref(math_mode)),
            "cublasGetMathMode",
        )
        self._check(
            self.library.cublasGetPointerMode_v2(
                self.handle, ctypes.byref(pointer_mode)
            ),
            "cublasGetPointerMode_v2",
        )
        if math_mode.value != CUBLAS_PEDANTIC_MATH:
            raise RuntimeError(f"cuBLAS math mode is not pedantic: {math_mode.value}")
        if pointer_mode.value != CUBLAS_POINTER_MODE_HOST:
            raise RuntimeError(f"cuBLAS pointer mode is not host: {pointer_mode.value}")
        self.version = version.value
        self.math_mode = math_mode.value
        self.pointer_mode = pointer_mode.value

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != CUBLAS_STATUS_SUCCESS:
            raise RuntimeError(f"{operation} failed with cuBLAS status {status}")

    def gemm(self, query: torch.Tensor, base: torch.Tensor, output: torch.Tensor) -> None:
        assert not self.closed and int(torch.cuda.current_stream().cuda_stream) == self.stream
        if query.dtype != torch.float32 or base.dtype != torch.float32:
            raise TypeError("pedantic GEMM requires FP32 input")
        if not query.is_contiguous() or not base.is_contiguous() or not output.is_contiguous():
            raise RuntimeError("pedantic GEMM requires contiguous tensors")
        m, k = map(int, query.shape)
        n, base_k = map(int, base.shape)
        if base_k != k or output.numel() != m * n:
            raise RuntimeError((query.shape, base.shape, output.shape))
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        # Row-major output [M,N] shares memory with column-major C^T [N,M].
        # B(row N,K) is column-major B^T(K,N), hence op(B^T)=B via transa=T.
        self._check(
            self.library.cublasGemmEx(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                m,
                k,
                ctypes.byref(alpha),
                ctypes.c_void_p(base.data_ptr()),
                CUDA_R_32F,
                k,
                ctypes.c_void_p(query.data_ptr()),
                CUDA_R_32F,
                k,
                ctypes.byref(beta),
                ctypes.c_void_p(output.data_ptr()),
                CUDA_R_32F,
                n,
                CUBLAS_COMPUTE_32F_PEDANTIC,
                CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmEx",
        )

    def record(self) -> dict[str, object]:
        return {
            "library": str(self.library_path),
            "library_sha256": sha256_file(self.library_path),
            "version": self.version,
            "handle": self.handle_value,
            "ownership": "engine-owned",
            "workspace_bytes": self.workspace_bytes,
            "stream": self.stream,
            "math_mode": self.math_mode,
            "required_math_mode": CUBLAS_PEDANTIC_MATH,
            "pointer_mode": self.pointer_mode,
            "compute_type": CUBLAS_COMPUTE_32F_PEDANTIC,
            "algorithm": CUBLAS_GEMM_DEFAULT,
            "input_output_type": CUDA_R_32F,
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        }

    def close(self):
        assert not self.closed
        torch.cuda.synchronize()
        self._check(self.library.cublasDestroy_v2(self.handle), 'cublasDestroy_v2')
        self.closed = True
        self.workspace = None
