# Adapter examples

- `zs601_v3_adapter.py` connects the reusable diagnostic driver to `gaussian-splattingWithMask_v3`.
- Other projects should copy this file and replace camera loading, model loading and rendering calls while keeping the `render_validation` signature.
- Adapter output must stay inside the supplied `output_dir` and return a JSON-serializable dictionary.
