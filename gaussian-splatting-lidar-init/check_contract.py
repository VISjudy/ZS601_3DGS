"""Independent projection/encoding checks, runnable without CUDA or torch."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from colmap_io import View, read_views, rotation, encode_depth


def check(sparse=None):
    views = read_views(sparse) if sparse else []
    # Include an asymmetric principal point to catch transposition/sign mistakes.
    views += [View(1, 1, 'test.png', [1, 0, 0, 0], [0, 0, 0], 640, 480, 500, 530, 298, 237)]
    error = 0.0
    rng = np.random.default_rng(20260921)
    for v in views:
        pc = rng.uniform([-0.5, -0.4, 1], [0.5, 0.4, 6], (32, 3))
        pw = (pc - v.t) @ rotation(v.q)
        view, proj = v.matrices()
        clip = np.column_stack([pw, np.ones(len(pw))]) @ (proj @ view).T
        ndc = clip[:, :2] / clip[:, 3:4]
        cuda_pixel_index = (ndc + 1) * np.array([v.width, v.height])/2 - 0.5
        pinhole_center = pc[:, :2] / pc[:, 2:3] * [v.fx, v.fy] + [v.cx, v.cy]
        error = max(error, float(np.abs(cuda_pixel_index + 0.5 - pinhole_center).max()))
    assert error < 1e-9, error
    depths = np.array([[0, 0.001, 1.234, 65.535]], dtype=np.float64)
    encoded = encode_depth(depths, depths > 0)
    assert encoded.tolist() == [[0, 1, 1234, 65535]]
    try:
        encode_depth(np.array([65.536]), np.array([True]))
    except ValueError:
        pass
    else:
        raise AssertionError('Overflow must not silently clip')
    # In-memory PNG roundtrip avoids test-file cleanup or replacing user files.
    import io
    data = io.BytesIO()
    Image.fromarray(encoded).save(data, format='PNG')
    raw = data.getvalue()
    assert raw[24:26] == bytes([16, 0])
    assert np.array_equal(np.array(Image.open(io.BytesIO(raw))), encoded)
    print({'camera_count': len(views)-1, 'max_projection_error_px': error, 'uint16_mm_roundtrip': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sparse')
    check(p.parse_args().sparse)
