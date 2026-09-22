"""Read-only frozen-scene material audit; create new semantic provenance files."""
from pathlib import Path
import hashlib
import json
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DATA = PROJECT / 'scenes/zs601-meetingroom/synthetic-training-v001'
DEST = PROJECT / '3dgsResult/zs601-mesh-noglass-v006'
assert Path(bpy.data.filepath).resolve() == (DATA / 'source/scene.blend').resolve()
assert hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest() == 'e27c037aa4b5937d4f2a586638539d22277d9ef689657e93c41a7909962d887a'
assert bpy.context.scene.unit_settings.system == 'METRIC'
assert bpy.context.scene.unit_settings.scale_length == 1
DEST.mkdir(exist_ok=True)
OUT = DEST / 'audit'
OUT.mkdir(exist_ok=True)
assert not any(OUT.iterdir()), 'Audit output directory must be empty'
geom = np.load(DATA / 'source/geometry.npz')
parts = json.loads((DATA / 'source/geometry.json').read_text())['parts']
objects = sorted((o for o in bpy.context.scene.objects if o.type == 'MESH' and not o.hide_render), key=lambda o: o.name)
assert [o.name for o in objects] == [p['name'] for p in parts]

def props(block):
    return {str(k): str(block[k]) for k in block.keys() if k != '_RNA_UI'}

def material_record(mat):
    if mat is None:
        return dict(name=None, glass=False, custom_properties={}, active_nodes=[])
    active = set()
    if mat.use_nodes:
        stack = [n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
        while stack:
            n = stack.pop()
            if n in active:
                continue
            active.add(n)
            sockets = [n.inputs.get('Surface')] if n.type == 'OUTPUT_MATERIAL' else n.inputs
            for socket in sockets:
                if socket:
                    stack.extend(link.from_node for link in socket.links)
    records = []
    glass = False
    unresolved = False
    for node in sorted(active, key=lambda n: n.name):
        rec = dict(name=node.name, type=node.type)
        for socket_name in ['Transmission Weight', 'Transmission', 'IOR', 'Alpha', 'Roughness', 'Base Color', 'Color', 'Fac']:
            socket = node.inputs.get(socket_name)
            if socket is not None and hasattr(socket, 'default_value'):
                v = socket.default_value
                rec[socket_name] = dict(value=float(v) if isinstance(v, (int, float)) else list(v), linked=socket.is_linked)
        if node.type == 'BSDF_GLASS':
            glass = True
        if node.type == 'BSDF_PRINCIPLED':
            socket = node.inputs.get('Transmission Weight') or node.inputs.get('Transmission')
            if socket:
                glass |= float(socket.default_value) > .01
                unresolved |= socket.is_linked
        unresolved |= node.type in ['GROUP', 'BSDF_TRANSPARENT', 'MIX_SHADER', 'ADD_SHADER']
        records.append(rec)
    return dict(name=mat.name, glass=bool(glass), unresolved=bool(unresolved), custom_properties=props(mat), active_nodes=records,
                label_name_hint=any(s in mat.name.lower() for s in ['glass', 'transparent', 'glazing']))

materials = {}
face_glass = np.zeros(len(geom['faces']), dtype=bool)
face_material = np.full(len(face_glass), -1, dtype=np.int32)
rows = []
dg = bpy.context.evaluated_depsgraph_get()
for obj, part in zip(objects, parts):
    ev = obj.evaluated_get(dg)
    mesh = ev.to_mesh()
    mesh.calc_loop_triangles()
    verts = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get('co', verts.ravel())
    matrix = np.array(obj.matrix_world)
    verts = (verts @ matrix[:3, :3].T + matrix[:3, 3]).astype('f4')
    faces = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype='i4') + part['v0']
    assert np.array_equal(verts, geom['vertices'][part['v0']:part['v0'] + part['nv']]), obj.name
    assert np.array_equal(faces, geom['faces'][part['f0']:part['f0'] + part['nf']]), obj.name
    slots = []
    for slot in obj.material_slots:
        name = slot.material.name if slot.material else '__NONE__'
        if name not in materials:
            record = material_record(slot.material)
            record['id'] = len(materials)
            materials[name] = record
        slots.append(materials[name])
    for j, tri in enumerate(mesh.loop_triangles):
        mi = mesh.polygons[tri.polygon_index].material_index
        if mi < len(slots):
            face_glass[part['f0'] + j] = slots[mi]['glass']
            face_material[part['f0'] + j] = slots[mi]['id']
    ng = int(face_glass[part['f0']:part['f0'] + part['nf']].sum())
    rows.append({**part, 'custom_properties': props(obj), 'materials': [m['name'] for m in slots], 'glass_triangles': ng})
    ev.to_mesh_clear()
np.save(OUT / 'glass_face_mask.npy', face_glass, allow_pickle=False)
np.save(OUT / 'face_material_id.npy', face_material, allow_pickle=False)
report = dict(source_scene=str(DATA / 'source/scene.blend'), source_scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
              blender_version=bpy.app.version_string, units='metres', source_geometry_exact=True,
              classification='Active Glass BSDF or constant active Principled Transmission Weight >0.01. Material names and custom properties recorded as secondary evidence. Complex/linked shaders require review.',
              materials=list(materials.values()), parts=rows, glass_triangles=int(face_glass.sum()),
              glass_parts=[r['name'] for r in rows if r['glass_triangles']],
              review_materials=[r['name'] for r in materials.values() if r.get('unresolved') or r.get('label_name_hint') and not r['glass']])
with (OUT / 'material_audit.json').open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('GLASS_AUDIT', json.dumps(dict(glass_parts=report['glass_parts'], glass_triangles=report['glass_triangles'],
      glass_materials=[m for m in materials.values() if m['glass']], review_materials=report['review_materials']), ensure_ascii=False), flush=True)
