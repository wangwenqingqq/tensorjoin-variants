# Evidence schema note

Some new diagnostic writers initialize a `pass_` placeholder using Python's
`dict(pass_=False, ...)` and later write their final outcome to the separate
JSON key `pass`. This unused initial placeholder is not updated. It is retained
rather than retrospectively editing raw outputs. For these qualification and
ABI writers the final `pass` key, exception fields, process exit code and guard
outcome jointly determine admission. Trace, numeric-sample, Graph and offline
subrecords that use only `pass_` instead set it to True after all assertions.

Admission must read `pass` when present, otherwise require `pass_ is True`, and
must independently require guard success and zero child exit code for GPU runs.
A False final `pass` cannot be overridden by a True `pass_`.
