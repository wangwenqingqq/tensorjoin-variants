# G11 Download Transport Note

The official HTTPS archive advertised 83058288 bytes. The initial local curl
download exited with code 18 after receiving 60522496 bytes, reporting
22535792 bytes remaining. No valid scientific input was accepted at that point.

The next request used `curl -C -` against the same URL and partial file;
`g11_download_resume_headers.txt` records its Content-Range. The completed
archive is 83058288 bytes, SHA-256
`49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276`.
Python zipfile's full CRC test passed before the decision. The empty resume
stderr log is retained. This was one logical data acquisition with a transport
resume, not a change of dataset or experimental sampling.
