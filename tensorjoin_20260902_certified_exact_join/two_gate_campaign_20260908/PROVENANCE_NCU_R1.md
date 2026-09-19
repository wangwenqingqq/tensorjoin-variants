# NCU/cuobjdump comparable body normalization

The first byte-for-byte text comparison failed before timing. For the original
F8 function, both exports contained exactly 4,232 instructions. Differences were
limited to NCU omitting `.reuse` operand hints, displaying the same LDGSTS
operands in a different order, and decorating one BRA target with a local label.
No kernel changed. The failed comparison and raw exports are preserved.

r1 defines a narrowly scoped comparable-body hash: relative PCs and branch/
reconvergence targets, omit unavailable `.reuse` hints, reorder NCU's three
LDGSTS display operands into cuobjdump's order, and remove only decorative
`(.L_x_N) branch labels while retaining their numeric target. No register or
operand value, predicate, opcode modifier or target is deleted.

This comparable hash cannot independently verify omitted operand-reuse hints.
The full cubin hashes, original static normalized hashes (with hints), frozen
Driver/JIT launch records and raw NCU source are retained as separate evidence.
Actual programs are directly bound to those code objects, not selected by an
unresolved library dispatcher. SourceCounters are mechanism evidence, not
performance measurements or a universal numerical guarantee.
