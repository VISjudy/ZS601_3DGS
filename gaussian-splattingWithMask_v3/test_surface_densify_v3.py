import unittest
from types import SimpleNamespace
import torch

from surface_densify_v3 import select_surface_candidates,make_surface_children
from render_v3 import rank_worst_test


class SurfaceDensifyTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(surface_densify_grad_threshold=.2,
            surface_densify_min_opacity=.01,surface_densify_min_views=20,
            surface_densify_max_children_per_seed=2,
            surface_densify_plane_ratio=.25,tangent_radius_ratio=1.,
            surface_densify_offset_ratio=.35,surface_densify_child_scale=.7,
            size_ratio=2.,thickness_ratio=.1)

    def reference(self):
        return {'anchor':torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),
            'plane':torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),
            'normal':torch.tensor([[0.,0.,1.],[0.,0.,1.],[0.,0.,1.]]),
            'spacing':torch.ones(3),'confidence':torch.tensor([1.,0.,1.]),
            'source_id':torch.arange(3)}

    def test_candidate_requires_reliable_near_surface_point(self):
        xyz=torch.tensor([[0.,0.,.1],[1.,0.,0.],[2.,0.,.5]])
        state={'recent_epoch_views':torch.tensor([20,20,20]),'epoch_views':torch.zeros(3,dtype=torch.int32),
               'is_seed':torch.ones(3,dtype=torch.bool),'densify_count':torch.zeros(3,dtype=torch.int32)}
        mask=select_surface_candidates(xyz,torch.full((3,1),.1),self.reference(),state,
                                       torch.tensor([.3,.3,.3]),self.args())
        self.assertEqual(mask.tolist(),[True,False,False])

    def test_children_stay_on_plane_and_within_scale_caps(self):
        ref=self.reference(); state={'densify_count':torch.zeros(3,dtype=torch.int32)}
        xyz=ref['anchor'].clone(); scales=torch.tensor([[3.,3.,1.]]*3)
        rotation=torch.tensor([[1.,0.,0.,0.]]*3); opacity=torch.full((3,1),.2)
        child_xyz,log_scale,child_rotation,child_opacity=make_surface_children(
            xyz,scales,rotation,opacity,ref,state,torch.tensor([0,2]),self.args())
        self.assertTrue(torch.allclose(child_xyz[:,2],torch.zeros(2),atol=1e-7))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(child_xyz-ref['anchor'][[0,2]],dim=1),
                                       torch.full((2,),.35),atol=1e-6))
        active=log_scale.exp()
        self.assertTrue(bool((active[:,:2]<=2.).all()))
        self.assertTrue(bool((active[:,2]<=.1).all()))
        q=torch.sigmoid(child_opacity)
        self.assertTrue(torch.allclose(1-(1-q).square(),torch.full((2,1),.2),atol=1e-6))
        self.assertTrue(torch.equal(child_rotation,rotation[[0,2]]))

    def test_worst_ranking_is_deterministic(self):
        rows=[{'image_name':'b','masked_psnr':10.},
              {'image_name':'c','masked_psnr':9.},
              {'image_name':'a','masked_psnr':10.}]
        ranked=rank_worst_test(rows,2)
        self.assertEqual([r['image_name'] for r in ranked],['c','a'])


if __name__=='__main__':
    unittest.main()
