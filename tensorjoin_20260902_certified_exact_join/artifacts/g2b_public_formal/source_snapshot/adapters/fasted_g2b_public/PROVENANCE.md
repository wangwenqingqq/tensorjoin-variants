# FaSTED G2B private evaluation adapter

This directory was derived from the source archive
`artifacts/upstream/fasted_9af85ed8_source.tar.gz`, pinned at revision
`9af85ed8edc818d5aa7cc0473187631adab3e6ba` (tree
`cb1b71b2b56ff59865c18211957d7fcc9781e3da`).

No license file was present at that pinned revision. This adapter is retained
only for private reproducibility/evaluation and must not be redistributed.

The search kernel and distance predicate are unchanged. Adapter-only changes:

1. read the frozen raw float32 source into pageable host memory;
2. time FP16 conversion/padding through sorted canonical host output;
3. retain the upstream D2H pair vector instead of discarding it;
4. serialize canonical IDs after the public timer for quality auditing.

