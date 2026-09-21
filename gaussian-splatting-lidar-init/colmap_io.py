"""Camera-only COLMAP TXT input. World-to-camera, Hamilton wxyz, metres.

Pixel coordinates follow COLMAP (top-left pixel centre is 0.5, 0.5).
No GT image is opened by this module or by the renderer.
"""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import json
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-5:
        raise ValueError('Invalid COLMAP unit quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass
class View:
    image_id: int
    camera_id: int
    name: str
    q: list
    t: list
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrices(self, near=0.01, far=100.0):
        view = np.eye(4)
        view[:3, :3] = rotation(self.q)
        view[:3, 3] = self.t
        proj = np.zeros((4, 4))
        proj[0, 0] = 2 * self.fx / self.width
        proj[1, 1] = 2 * self.fy / self.height
        proj[0, 2] = 2 * self.cx / self.width - 1
        proj[1, 2] = 2 * self.cy / self.height - 1
        proj[2, 2] = far / (far-near)
        proj[2, 3] = -far * near / (far-near)
        proj[3, 2] = 1
        return view, proj


def read_views(sparse):
    sparse = Path(sparse)
    cameras = {}
    for line in (sparse / 'cameras.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        a = line.split()
        cid, model, width, height = int(a[0]), a[1], int(a[2]), int(a[3])
        if model == 'PINHOLE':
            fx, fy, cx, cy = map(float, a[4:])
        elif model == 'SIMPLE_PINHOLE':
            f, cx, cy = map(float, a[4:])
            fx = fy = f
        else:
            raise ValueError('Only undistorted PINHOLE/SIMPLE_PINHOLE supported: ' + model)
        if cid in cameras or width <= 0 or height <= 0 or min(fx, fy) <= 0:
            raise ValueError('Invalid or duplicate camera')
        cameras[cid] = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    views = []
    with (sparse / 'images.txt').open(encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            a = line.split(maxsplit=9)
            if len(a) != 10:
                raise ValueError('Malformed COLMAP image line')
            iid, cid = int(a[0]), int(a[8])
            name = a[9].strip()
            if PurePosixPath(name).name != name or '\\' in name or not name.lower().endswith('.png'):
                raise ValueError('Expected flat PNG names, no path traversal')
            v = View(iid, cid, name, list(map(float, a[1:5])), list(map(float, a[5:8])), **cameras[cid])
            rotation(v.q)
            if not np.isfinite(v.t).all():
                raise ValueError('Invalid translation')
            views.append(v)
            if f.readline() == '':
                raise ValueError('Missing COLMAP POINTS2D line')
    if not views or len({v.image_id for v in views}) != len(views) or len({v.name for v in views}) != len(views):
        raise ValueError('Empty/duplicate views')
    return sorted(views, key=lambda v: v.image_id)


def write_sparse(sparse, views, xyz, rgb):
    sparse = Path(sparse)
    sparse.mkdir(parents=True, exist_ok=False)
    cameras = {v.camera_id: v for v in views}
    with (sparse / 'cameras.txt').open('x') as f:
        f.write('# CAMERA_ID MODEL WIDTH HEIGHT FX FY CX CY\n')
        for cid, v in sorted(cameras.items()):
            f.write(f'{cid} PINHOLE {v.width} {v.height} {v.fx:.17g} {v.fy:.17g} {v.cx:.17g} {v.cy:.17g}\n')
    with (sparse / 'images.txt').open('x') as f:
        f.write('# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# Empty POINTS2D lines: render cameras, no feature tracks.\n')
        for v in views:
            nums = ' '.join(format(x, '.17g') for x in v.q + v.t)
            f.write(f'{v.image_id} {nums} {v.camera_id} {v.name}\n\n')
    with (sparse / 'points3D.txt').open('x') as f:
        f.write('# POINT3D_ID X Y Z R G B ERROR TRACK[]\n# Initial input cloud. No SfM observations or reprojection residuals; ERROR=0 placeholder.\n')
        for i, (p, c) in enumerate(zip(xyz, rgb), 1):
            f.write(f'{i} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {int(c[0])} {int(c[1])} {int(c[2])} 0\n')


def encode_depth(depth_m, valid):
    depth_m = np.asarray(depth_m)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth_m) & (depth_m > 0)
    if np.any(valid & (depth_m > 65.535)):
        raise ValueError('Depth exceeds uint16 millimetre range; refuse clipping')
    out = np.zeros(depth_m.shape, dtype=np.uint16)
    out[valid] = np.maximum(1, np.rint(depth_m[valid] * 1000)).astype(np.uint16)
    return out
