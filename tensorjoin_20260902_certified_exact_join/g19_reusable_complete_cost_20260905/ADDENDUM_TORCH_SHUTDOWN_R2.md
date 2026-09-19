# G19 R2: explicit cached-allocator shutdown, unchanged public operation

R1 RT full memcheck now reports zero errors and zero leaked bytes; its synccheck
also passes. Keep those gates and the R1 binaries. The subsequent TC full
memcheck returns all four correct outputs but reports six unfreed allocations
at process exit, so R1 remains failed and no public timing is admitted.

## Observed first boundary

Five allocation stacks originate in PyTorch's device caching allocator. Their
sum is 425,721,856 bytes, exactly the reported reserved device cache after every
call and after the old engine close; live tensor allocation is zero throughout.
The sixth is a 4-byte cudaHostAlloc from the pinned caching host allocator used
by scalar metadata readback. These are distinct from the fixed OWL wrong-free
bug. Do not call the new leak gate clean merely because these appear to be caches.

The pre-existing FP32 A0 matrix separately records a constant 8,519,680 live
allocator bytes after library preparation and every call. The expected owner
is the borrowed PyTorch cuBLAS workspace. R2 will record the before/after state
when explicitly clearing library workspaces, then unused device and pinned caches.
No borrowed cuBLAS handle is manually destroyed.

## Explicit end-of-engine cleanup

In a separate operators_r2 module, after the last output is consumed and streams
are synchronized, clear this process's PyTorch cuBLAS workspaces, call
torch.cuda.empty_cache(), and call the installed ATen host allocator's
empty_cache() through a tiny typed C ABI bridge. These APIs are verified in the
installed 2.11 headers/stubs. Assert device live/reserved bytes and host current
allocation state after shutdown, and retain the staged memory receipt. Full
Compute Sanitizer leak checking remains enabled; any remaining error fails.

This happens ONLY at engine destruction, outside the already declared warmed
per-operation denominator. No cache flush, buffer bypass, changed kernel, input
metadata, output or parameter is inserted between timed calls. Code/libraries
and all source hashes remain separately frozen. The R1 failed shutdown stays
failed even if R2 succeeds.

## Stress proxy correction before stress starts

The protocol requires no live allocation accumulation, not zero bytes while an
explicitly retained library workspace is alive. The old runner's unconditional
zero assertion would incorrectly reject the already observed constant FP32
workspace. R2 compares each post-call live allocation to the recorded
after-preparation baseline (0 for TC, observed 8,519,680 for FP32), and still
requires zero after workspace/cache destruction. The original device/RSS growth
bounds, call counts and exact-ID checks are unchanged. This exception is explicit
before any stress or latency sample; there is no failed stress observation to hide.

Retain six already admitted R1-composite slots (four matrices and two RT safety
slots), rerun failed TC memcheck and execute the remaining pending slots with
R2 closure. Revalidate source/code identity. The allocator bridge is not a new
join algorithm or paper contribution.
