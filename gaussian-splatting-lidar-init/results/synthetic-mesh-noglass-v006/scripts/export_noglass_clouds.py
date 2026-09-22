"""Create two mesh-derived, glass-free colored PLYs with explicit provenance.

Reuse exact preexisting 1 cm surface samples and train-only colors for a controlled
glass ablation. The 3 cm cloud is a per-component voxel subset, not averaged XYZ.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
os.environ['OPENBLAS_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT / '.runtime/synthetic-dataset'))
sys.path.insert(0, str(PROJECT / 'experiments/2026-09-21/synthetic-lidar-init-v001/python-deps'))
import numpy as np
from plyfile import PlyData, PlyElement

DATA = PROJECT / 'scenes/zs601-meetingroom/synthetic-training-v001'
OLD = PROJECT / '3dgsResult/zs601-mesh1cm-scale1-v003/cloud'
OUT = PROJECT / '3dgsResult/zs601-mesh-noglass-v006'
AUDIT = OUT / 'audit'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def save(p, data):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

t0 = time.time()
audit = json.loads((AUDIT / 'material_audit.json').read_text(encoding='utf-8'))
assert audit['source_geometry_exact']
assert sha(DATA / 'source/scene.blend') == audit['source_scene_sha256']
source_sha = sha(OLD / 'points_mesh_1cm.ply')
assert source_sha == 'f7f822cc86ca6af3e376f9cbc902aeff6ae3537724d2c5b7ed8bc5a5da63a8eb'
source = PlyData.read(OLD / 'points_mesh_1cm.ply')['vertex'].data
face_ids = np.load(OLD / 'mesh_face_id.npy', mmap_mode='r')
view_ids = np.load(OLD / 'color_source_view_id.npy', mmap_mode='r')
material_ids = np.load(AUDIT / 'face_material_id.npy')
selected = [m for m in audit['materials'] if m['custom_properties'].get('semantic_class') == 'clear_glass']
assert len(selected) == 1
for m in selected:
    assert not m['unresolved']
    assert any(n.get('Transmission Weight', {}).get('value') == 1.0 for n in m['active_nodes'])
excluded_faces = np.isin(material_ids, [m['id'] for m in selected])
np.save(AUDIT / 'excluded_clear_glass_faces.npy', excluded_faces, allow_pickle=False)
parts = audit['parts']
face_objects = np.empty(len(material_ids), dtype='i4')
for i, p in enumerate(parts):
    face_objects[p['f0']:p['f0'] + p['nf']] = i
excluded_parts = [p['name'] for p in parts if excluded_faces[p['f0']:p['f0']+p['nf']].any()]
assert len(excluded_parts) == 33
assert len(source) == len(face_ids) == len(view_ids) == 7180820
keep1 = np.flatnonzero(~excluded_faces[face_ids])
assert len(keep1) > 6000000
decision = dict(criterion='Explicit material semantic_class=clear_glass, verified constant Principled Transmission Weight=1',
                excluded_materials=[m['name'] for m in selected], excluded_parts=excluded_parts,
                excluded_triangles=int(excluded_faces.sum()), excluded_source_points=int(len(source)-len(keep1)),
                retained_transmissive_plastics=['ZS601_LowV36_WhitePolyethylene','ZS601_Semantic_FrostedOrganizerPlastic'],
                retained_opaque_glass_named_materials=audit['review_materials'],
                preliminary_audit_note='material_audit.json glass flag and glass_face_mask.npy denote broad transmission candidates; only excluded_clear_glass_faces.npy is the approved final selection.',
                semantic_image_masks_added=False)
save(AUDIT / 'semantic_decision.json', decision)
xyz = np.column_stack([source[k] for k in ['x','y','z']])
object_ids = face_objects[face_ids]
xyz1 = xyz[keep1].astype('f8')
ijk = np.floor(xyz1/.03).astype('i8')
shift = ijk + 4096
assert ((shift >= 0) & (shift < 8192)).all()
key = ((object_ids[keep1].astype('i8')*8192 + shift[:,0])*8192 + shift[:,1])*8192 + shift[:,2]
score = np.square(xyz1-(ijk+.5)*.03).sum(1)
order = np.lexsort((score,key))
keep3 = np.sort(keep1[order[np.r_[True,key[order][1:] != key[order][:-1]]]])
del xyz1, ijk, shift, key, score, order
manifest = dict(status='EXPORTED_PENDING_INDEPENDENT_VALIDATION', source_cloud_sha256=source_sha,
                source_scene_sha256=audit['source_scene_sha256'], units='metres',
                semantic_selection=decision, original_points=len(source), outputs={},
                color_policy='Exact RGB and normals from prior 1cm mesh samples. Train-only projected colors and material fallback retained; no recoloring or geometry changes beyond excluded glass.',
                color_limit='Unobserved and glass-occluded opaque surfaces may retain constant material fallback; removing glass alone does not bake their textures, reflections or lighting.',
                point_spacing='Per-component world-axis voxel resolution; one actual surface representative per occupied cell. Not a Poisson disk or strict minimum-distance guarantee.',
                new_formal_training_init='cloud_3cm/points_mesh_3cm_noglass.ply', new_virtual_view_input='cloud_1cm/points_mesh_1cm_noglass.ply',
                source_las_used=False, source_las_note='User visual observation motivated this approximation; this is full mesh surface sampling, not a physical LiDAR return simulation.')
for label, spacing, indices in [('1cm', .01, keep1), ('3cm', .03, keep3)]:
    folder = OUT / ('cloud_' + label)
    folder.mkdir(exist_ok=False)
    path = folder / ('points_mesh_' + label + '_noglass.ply')
    arr = np.array(source[indices], copy=True)
    with path.open('xb') as f:
        PlyData([PlyElement.describe(arr, 'vertex')], text=False,
                comments=['units metres; original mesh surface representatives', 'known clear_glass material excluded',
                          label + ' world voxels per component; RGB unchanged from train-only baseline']).write(f)
    np.save(folder / 'source_sample_id.npy', indices.astype('u4'), allow_pickle=False)
    np.save(folder / 'mesh_face_id.npy', face_ids[indices], allow_pickle=False)
    np.save(folder / 'color_source_view_id.npy', view_ids[indices], allow_pickle=False)
    manifest['outputs'][label] = dict(path=path.relative_to(OUT).as_posix(), points=len(indices), voxel_size_m=spacing,
        components=int(np.unique(object_ids[indices]).size), bounds=[xyz[indices].min(0).tolist(),xyz[indices].max(0).tolist()],
        projected_points=int((view_ids[indices]>=0).sum()), fallback_points=int((view_ids[indices]<0).sum()),
        excluded_material_points=int(excluded_faces[face_ids[indices]].sum()),
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir() if p.is_file()})
    print('CLOUD_EXPORTED',label,json.dumps(manifest['outputs'][label]),flush=True)
    del arr
assert sha(OLD / 'points_mesh_1cm.ply') == source_sha
manifest['seconds'] = time.time()-t0
save(OUT / 'cloud_manifest.json', manifest)
print('TWO_CLOUDS_EXPORTED', json.dumps({k:v['points'] for k,v in manifest['outputs'].items()}),flush=True)
