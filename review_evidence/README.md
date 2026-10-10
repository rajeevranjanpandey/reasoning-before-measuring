# Complete research archive parts

21 binary parts reconstruct the three original ZIPs without changing their bytes. Each part is at most 8 MiB, allowing repository upload and inclusion in the anonymous snapshot.

Download the whole repository, then run `python assemble_review_archives.py` from its root. The assembler verifies the reconstructed archives against `RELEASE_SHA256.json`. Extract all three ZIPs into one empty directory and verify the 155 research files against `MANIFEST_SHA256.json`.

No PDF, LaTeX or GGUF files are included in the research archives. The model-download launcher is a separate delivery addition; recorded research scripts and outcomes are unchanged.
