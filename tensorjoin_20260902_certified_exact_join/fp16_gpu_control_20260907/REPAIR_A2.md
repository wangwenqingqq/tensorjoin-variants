# Harness A2 repairs; unchanged numerical programs

GPU0 fixture_a1 completed its first131,328-pair panel:796accepted,
43stage1-ambiguous,40FP64-input pairs, exact output and interval checks passed.
The process then failed in artifact metadata serialization: Triton GPUTarget is
not a JSON-native type. This run remains failed, not admitted. ControlR2 adds
stable string serialization for that compiler target, retaining all other
metadata fields and actual cubin/PTX identities. ValidatorR2 uses new result
labels. The metadata/classifier source and launch contract are unchanged.

The initial offline NCU normalizer failed its cross-launch identity assertion
because it did not relocate BSSY reconvergence PCs. The two diffs each contain
only that absolute loaded address. The R1 normalizer adds BSSY targets to the
explicit branch relocation set and writes separate normalized artifacts; the
original listings remain retained. No register, constant, or opcode removal.
