# Source byte provenance

`drive_source_manifest.json` hashes the original Drive downloads before Git line
ending normalization. The downloaded Python sources use CRLF. Git stores the
working Python modules as LF; this changes their byte hashes, not their code.

The executed archive from commit `4c7186e363f14050c4977bb192f12ed237112764`
is separately hashed by `source_manifest.json` in the delivered reproducibility
folder. Its unused `render_from_sparse_v4.original.py` reference was also stored
as LF (SHA256 `47adad4399773db28b7348a7c9227a439be1d7ea0e82cdebdb13640a3a8ecf97`).

The final branch preserves that reference file byte-for-byte as the Drive CRLF
download (SHA256 `8a74e8b175cf3173d6572c6959e192ef1f766f7ab48d706d528f0a2e96ade490`),
using its `-text` attribute. Only this unused reference's line endings changed
after execution; runtime code and CUDA math are unchanged.
