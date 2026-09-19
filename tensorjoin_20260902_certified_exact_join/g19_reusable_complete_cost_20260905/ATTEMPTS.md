# G19 attempts (append only)

- 2026-09-05 prebuild generation a0: exact source-replacement assertion rejected
  an incorrect spelling of G18's counter initializer. No build or GPU launch.
  The partial tree and generator are retained in artifacts/rejected_generation_a0.
  Generation r1 corrects the expected initializer only; the protocol is unchanged.
- Initial task-owned SSH setup was closed by the jump host twice. A later shell
  without a local PTY could not accept input; the next explicit PTY/multiplex
  shell succeeded. Transport only; no experiment or global SSH settings changed.
