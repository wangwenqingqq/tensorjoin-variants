# R3 pre-launch synchronization recovery

A broad source-directory upload temporarily removed one final blank line from
run_slot_r1.py and run_slot_r2.py because the local copies lagged the remote
frozen versions. The immutable R1 hash gate aborted before any GPU job.
Restored the exact bytes (one LF per file); expected SHA-256 values matched
without modifying either manifest. All A0/R1/R2 frozen hashes must pass again.
Subsequent uploads use exact new file targets, not whole source directories.
No result or accepted source semantics changed; the failed freeze attempt is retained.
