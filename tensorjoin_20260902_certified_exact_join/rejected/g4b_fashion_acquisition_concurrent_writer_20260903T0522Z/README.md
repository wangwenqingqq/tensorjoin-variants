# Rejected Fashion-MNIST acquisition attempt

The first SSH/PTTY interruption did not terminate its remote curl. A second
resume therefore created two writers for the same partial file. Both exact curl
PIDs were terminated by this task. The partial file is invalid, is retained
only as negative operational evidence, and must not be used as a dataset.
Acquisition restarts from byte zero using the official Fashion-MNIST S3 source.
