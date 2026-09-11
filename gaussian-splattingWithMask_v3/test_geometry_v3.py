"""Run: python -m unittest -v test_geometry_v3 (CPU, no CUDA extensions)."""
import unittest
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from geometry_v3 import normals_to_quaternions,normal_axis,geometry_losses,build_reference
from arguments_v3 import parse_args,FEATURES,LOSSES
from runtime_v3 import fresh_topology,finish_epoch,prune

def config(experiment='B',extra=()):
    return parse_args(['--experiment',experiment,'-s','data','-m','output','--point_cloud','p.las',
        '--train_file','train.txt','--val_file','val.txt','--cameras_file','cameras.txt',*extra])[0]

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.r={'anchor':torch.zeros(1,3),'plane':torch.zeros(1,3),'normal':torch.tensor([[0.,0.,1.]]),
                'spacing':torch.ones(1),'confidence':torch.ones(1),'source_id':torch.tensor([0])}
    def test_rotation_full_coverage(self):
        rng=np.random.default_rng(42); n=rng.normal(size=(10000,3)); n/=np.linalg.norm(n,axis=1,keepdims=True)
        n=np.vstack([n,np.eye(3),-np.eye(3)])
        q=normals_to_quaternions(n)
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_allclose(np.linalg.norm(q,axis=1),1,atol=2e-7)
        np.testing.assert_allclose(normal_axis(torch.from_numpy(q)).numpy(),n,atol=6e-7)
    def test_invalid_normal_fallback(self):
        q=normals_to_quaternions([[0,0,0],[np.nan,0,0]])
        np.testing.assert_allclose(normal_axis(torch.from_numpy(q)),[[0,0,1],[0,0,1]],atol=1e-6)
    def test_preset_and_override(self):
        a=config('A'); b=config('B',['--normal_loss','off','--init_flatten','off'])
        self.assertTrue(a.init_flatten)
        self.assertFalse(a.surface_loss)
        self.assertTrue(b.surface_loss)
        self.assertFalse(b.normal_loss); self.assertFalse(b.init_flatten)
    def test_c_adds_only_scale_bounds_to_b_preset(self):
        b=config('B'); c=config('C')
        expected_b={name: name not in ('scale_bounds','surface_densify') for name in FEATURES}
        self.assertEqual({name:getattr(b,name) for name in FEATURES},expected_b)
        self.assertEqual(
            {name:getattr(c,name) for name in FEATURES if name!='scale_bounds'},
            {name:getattr(b,name) for name in FEATURES if name!='scale_bounds'})
        self.assertTrue(c.scale_bounds)
    def test_c_scale_bounds_single_override_off(self):
        c=config('C',['--scale_bounds','off'])
        self.assertFalse(c.scale_bounds)
        self.assertTrue(all(getattr(c,name) for name in FEATURES if name not in ('scale_bounds','surface_densify')))
        self.assertFalse(c.surface_densify)
    def test_d_adds_only_surface_densify_to_c_preset(self):
        c=config('C'); d=config('D')
        self.assertEqual(
            {name:getattr(d,name) for name in FEATURES if name!='surface_densify'},
            {name:getattr(c,name) for name in FEATURES if name!='surface_densify'})
        self.assertFalse(c.surface_densify)
        self.assertTrue(d.surface_densify)
    def test_d_surface_densify_single_override_off(self):
        c=config('C'); d=config('D',['--surface_densify','off'])
        self.assertEqual(
            {name:getattr(d,name) for name in FEATURES},
            {name:getattr(c,name) for name in FEATURES})
    def loss(self,name,xyz,scales,q=None):
        a=config('A',['--'+name+'_loss','on','--'+name+'_warmup','0'])
        q=torch.tensor([[1.,0.,0.,0.]],requires_grad=True) if q is None else q
        return geometry_losses(xyz,scales,q,self.r,a,1)[0]
    def test_surface_gradient(self):
        x=torch.tensor([[0.,0.,.1]],requires_grad=True)
        self.loss('surface',x,torch.ones(1,3)).backward()
        self.assertGreater(x.grad[0,2],0); self.assertEqual(x.grad[0,0],0)
    def test_tangent_deadzone(self):
        x=torch.tensor([[.2,0.,3.]],requires_grad=True)
        value=self.loss('tangent',x,torch.ones(1,3)); value.backward()
        self.assertEqual(value,0); self.assertEqual(x.grad.abs().sum(),0)
        x=torch.tensor([[2.,0.,0.]],requires_grad=True)
        self.loss('tangent',x,torch.ones(1,3)).backward(); self.assertGreater(x.grad[0,0],0)
    def test_flatten_only_third_axis(self):
        s=torch.tensor([[3.,3.,.2]],requires_grad=True)
        self.loss('flatten',torch.zeros(1,3),s).backward()
        self.assertEqual(s.grad[0,:2].abs().sum(),0); self.assertGreater(s.grad[0,2],0)
    def test_size_only_tangent(self):
        s=torch.tensor([[3.,1.,5.]],requires_grad=True)
        self.loss('size',torch.zeros(1,3),s).backward()
        self.assertGreater(s.grad[0,0],0); self.assertEqual(s.grad[0,2],0)
    def test_normal_sign_invariant_and_grad(self):
        q=torch.tensor([[.9238795,.3826834,0.,0.]],requires_grad=True)
        l=self.loss('normal',torch.zeros(1,3),torch.ones(1,3),q)
        self.assertGreater(l,0); l.backward(); self.assertTrue(torch.isfinite(q.grad).all())
        q2=torch.tensor([[0.,1.,0.,0.]])
        self.assertAlmostEqual(float(self.loss('normal',torch.zeros(1,3),torch.ones(1,3),q2)),0)
    def test_disabled_no_geometry_gradient(self):
        x=torch.zeros(1,3,requires_grad=True)
        l,terms=geometry_losses(x,torch.ones(1,3),torch.tensor([[1.,0.,0.,0.]]),self.r,config('A'),1)
        self.assertFalse(l.requires_grad); self.assertTrue(all(v['state']=='OFF' for v in terms.values()))
    def test_zero_confidence_safe(self):
        self.r['confidence'].zero_()
        self.assertEqual(self.loss('surface',torch.ones(1,3),torch.ones(1,3)),0)
    def test_camera_orientation(self):
        xx,yy=np.meshgrid(np.linspace(-1,1,15),np.linspace(-1,1,15))
        xyz=np.stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)],1)
        ref=build_reference(xyz,[[0,0,-2]],config())
        valid=ref['confidence']>0
        self.assertGreater(valid.mean(),.5)
        self.assertTrue((ref['normal'][valid,2]<-.99).all())
        np.testing.assert_allclose(ref['plane'][:,2],0,atol=1e-6)
    def test_prune_reference_alignment_without_densify(self):
        class Fake:
            def __init__(self):
                self.get_xyz=torch.zeros(100,3); self.get_opacity=torch.full((100,1),.1)
                self.get_opacity[5]=.001
            def prune_points(self,mask):
                assert self.tmp_radii.shape[0]==100
                self.get_xyz=self.get_xyz[~mask]; self.get_opacity=self.get_opacity[~mask]
        g=Fake(); state=fresh_topology(g); state['epoch_views'][:]=30
        a=config(); a.prune_patience=1
        r={'source_id':torch.arange(100)}
        r,state,event=prune(g,r,state,a,3000)
        self.assertEqual(event['removed'],1); self.assertNotIn(5,r['source_id'].tolist())
        self.assertEqual(len(state['epoch_views']),99)
        finish_epoch(state); self.assertEqual(state['epoch_views'].sum(),0)
        self.assertTrue((state['max_epoch_views']==30).all())
    def test_actual_baseline_prune_optimizer_state(self):
        # Exercise actual baseline methods without importing CUDA-only extensions.
        source=ast.parse((Path(__file__).parent/'scene/gaussian_model.py').read_text(encoding='utf-8'))
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_prune_optimizer','prune_points')]
        namespace={'torch':torch,'nn':torch.nn}
        exec(compile(ast.Module(body=methods,type_ignores=[]),'<baseline pruning>','exec'),namespace)
        g=type('Model',(),{n.name:namespace[n.name] for n in methods})()
        groups=[]
        for name,attr in [('xyz','_xyz'),('f_dc','_features_dc'),('f_rest','_features_rest'),
                          ('opacity','_opacity'),('scaling','_scaling'),('rotation','_rotation')]:
            param=torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(4,3))
            setattr(g,attr,param); groups.append({'params':[param],'name':name})
        g.optimizer=torch.optim.Adam(groups,lr=.01)
        sum(p.square().sum() for group in groups for p in group['params']).backward()
        g.optimizer.step(); g.optimizer.zero_grad()
        original=g.optimizer.state[g._xyz]['exp_avg'].clone()
        for name in ('xyz_gradient_accum','denom','max_radii2D','tmp_radii'):
            setattr(g,name,torch.arange(4))
        g.prune_points(torch.tensor([False,True,False,True]))
        torch.testing.assert_close(g.optimizer.state[g._xyz]['exp_avg'],original[[0,2]])
        self.assertEqual(g._xyz.shape,(2,3)); self.assertEqual(g.tmp_radii.tolist(),[0,2])

if __name__=='__main__': unittest.main()
