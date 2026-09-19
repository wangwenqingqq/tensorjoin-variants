The first static-audit helper recognized only hexadecimal FP64 immediates.
The compiler emitted the correct FP64 coefficients as decimal64-bit patterns
in mov.b64 instructions (4607182418800009216 and4607182418800021504).
The audit stopped before any sanitizer/timing launch. Extend its representation
parser to verify those exact bit patterns too; do not change kernel code or
relax numerical values. Original audit helper retained here.
