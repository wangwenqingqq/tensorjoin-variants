# Safety wrapper r1 (before sanitizer execution)

The inherited safety.py selected the same physical tiles but sorted them by ID,
so it would not exercise the new visit orders. safety_r1.py keeps the exact tile
set and filters each complete schedule to that set, retaining its visit order.
No sanitizer or performance observation existed before this correction. The
original frozen source remains unchanged; all sanitizer runs use safety_r1.py.
No CUDA source or generated program changes.
