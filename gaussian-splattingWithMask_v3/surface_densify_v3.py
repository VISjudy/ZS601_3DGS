"""D-only LiDAR-plane-aware controlled densification."""
import math
import torch
import torch.nn.functional as F


def accumulate_surface_gradient(g,pkg):
    """Accumulate the same screen-space position signal used by 3DGS, without growing yet."""
    points=pkg.get('viewspace_points')
    if points is None or points.grad is None:
        raise RuntimeError('Surface densification needs retained viewspace point gradients')
    visible=pkg['radii']>0
    grad=torch.linalg.vector_norm(points.grad[:,:2],dim=-1,keepdim=True)
    finite=visible[:,None]&torch.isfinite(grad)
    g.xyz_gradient_accum[finite[:,0]]+=grad[finite[:,0]]
    g.denom[finite[:,0]]+=1


def _tangent_basis(normal):
    normal=F.normalize(normal,dim=-1)
    ref=torch.zeros_like(normal); ref[:,2]=1
    ref[normal[:,2].abs()>.9]=normal.new_tensor([1.,0.,0.])
    tangent1=F.normalize(torch.cross(ref,normal,dim=-1),dim=-1)
    tangent2=torch.cross(normal,tangent1,dim=-1)
    return tangent1,tangent2


def select_surface_candidates(xyz,opacity,reference,state,gradient,a):
    """Return a boolean mask. Every condition has a direct geometry or evidence meaning."""
    n=reference['normal']; h=reference['spacing']; conf=reference['confidence']
    plane_distance=((xyz-reference['plane'])*n).sum(-1).abs()
    delta=xyz-reference['anchor']
    tangent=torch.linalg.vector_norm(delta-(delta*n).sum(-1,keepdim=True)*n,dim=-1)
    views=torch.maximum(state['epoch_views'],state['max_epoch_views'])
    return (torch.isfinite(gradient)&(gradient>=a.surface_densify_grad_threshold)&
            (opacity.flatten()>=a.surface_densify_min_opacity)&
            (conf>0)&(views>=a.surface_densify_min_views)&
            (plane_distance<=a.surface_densify_plane_ratio*h)&
            (tangent<=a.tangent_radius_ratio*h))


def make_surface_children(xyz,scales,rotation,opacity,reference,state,selected,a):
    n=F.normalize(reference['normal'][selected],dim=-1)
    h=reference['spacing'][selected]
    base=xyz[selected]
    plane=reference['plane'][selected]
    base=base-((base-plane)*n).sum(-1,keepdim=True)*n
    tangent1,tangent2=_tangent_basis(n)
    phase=state['densify_count'][selected].to(xyz.dtype)
    phase=phase+(reference['source_id'][selected]%8).to(xyz.dtype)
    angle=phase*(math.pi*(3-math.sqrt(5)))
    direction=torch.cos(angle)[:,None]*tangent1+torch.sin(angle)[:,None]*tangent2
    child_xyz=base+a.surface_densify_offset_ratio*h[:,None]*direction
    child_xyz=child_xyz-((child_xyz-plane)*n).sum(-1,keepdim=True)*n
    child_scale=scales[selected]*a.surface_densify_child_scale
    child_scale[:,:2]=torch.minimum(child_scale[:,:2],a.size_ratio*h[:,None])
    child_scale[:,2]=torch.minimum(child_scale[:,2],a.thickness_ratio*h)
    child_opacity=(opacity[selected]*.5).clamp(1e-4,.1)
    child_opacity=torch.log(child_opacity/(1-child_opacity))
    return child_xyz,child_scale.clamp_min(torch.finfo(scales.dtype).tiny).log(),rotation[selected],child_opacity


@torch.no_grad()
def surface_densify(g,reference,state,a,iteration):
    if (not a.surface_densify or iteration<a.surface_densify_start or
            iteration>a.surface_densify_until or iteration%a.surface_densify_interval):
        return reference,state,None
    if 'densify_count' not in state:
        state['densify_count']=torch.zeros(len(g.get_xyz),dtype=torch.int32,device=g.get_xyz.device)
    gradient=(g.xyz_gradient_accum/g.denom.clamp_min(1)).flatten()
    eligible=select_surface_candidates(g.get_xyz,g.get_opacity,reference,state,gradient,a)
    candidates=eligible.nonzero().flatten()
    initial_count=int(reference['source_id'].max().item())+1 if len(reference['source_id']) else 0
    max_points=max(initial_count,int(math.floor(initial_count*a.surface_densify_max_points_ratio)))
    budget=max(0,max_points-len(g.get_xyz))
    event_cap=max(1,int(math.floor(len(g.get_xyz)*a.surface_densify_max_fraction)))
    take=min(len(candidates),budget,event_cap)
    if take==0:
        g.xyz_gradient_accum.zero_(); g.denom.zero_()
        return reference,state,{'iteration':iteration,'eligible':len(candidates),'added':0,
            'count':len(g.get_xyz),'budget_remaining':budget,
            'criterion':'screen gradient + opacity + views + reliable LiDAR plane + surface/tangent gates'}
    order=torch.argsort(gradient[candidates],descending=True,stable=True)
    selected=candidates[order[:take]]
    child_xyz,child_scaling,child_rotation,child_opacity=make_surface_children(
        g.get_xyz,g.get_scaling,g._rotation,g.get_opacity,reference,state,selected,a)
    tensors={'xyz':child_xyz,'f_dc':g._features_dc[selected],
             'f_rest':g._features_rest[selected],'opacity':child_opacity,
             'scaling':child_scaling,'rotation':child_rotation}
    old_accum=g.xyz_gradient_accum; old_denom=g.denom; old_radii=g.max_radii2D
    optimized=g.cat_tensors_to_optimizer(tensors)
    g._xyz=optimized['xyz']; g._features_dc=optimized['f_dc']
    g._features_rest=optimized['f_rest']; g._opacity=optimized['opacity']
    g._scaling=optimized['scaling']; g._rotation=optimized['rotation']
    reference={k:torch.cat((v,v[selected]),dim=0) for k,v in reference.items()}
    state['densify_count'][selected]+=1
    state={k:torch.cat((v,torch.zeros(take,dtype=v.dtype,device=v.device)),dim=0)
           for k,v in state.items()}
    g.xyz_gradient_accum=torch.cat((old_accum,torch.zeros((take,1),device=old_accum.device)),dim=0)
    g.denom=torch.cat((old_denom,torch.zeros((take,1),device=old_denom.device)),dim=0)
    g.max_radii2D=torch.cat((old_radii,torch.zeros(take,device=old_radii.device)),dim=0)
    g.xyz_gradient_accum.zero_(); g.denom.zero_()
    return reference,state,{'iteration':iteration,'eligible':len(candidates),'added':take,
        'count':len(g.get_xyz),'budget_remaining':max_points-len(g.get_xyz),
        'mean_selected_gradient':float(gradient[selected].mean()),
        'offset_ratio':a.surface_densify_offset_ratio,
        'criterion':'screen gradient + opacity + views + reliable LiDAR plane + surface/tangent gates'}
