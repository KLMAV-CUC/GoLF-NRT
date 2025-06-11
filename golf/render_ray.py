import torch
from collections import OrderedDict

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################

# Kernel regression 
def sample_pdf_g(bins, weights, N_samples, det=True):
    def ker(x, y, h=0.03):   # kernek function
        x = x.unsqueeze(-1)  # Add a singleton dimension for broadcasting
        y = y.unsqueeze(-2)   # Add a singleton dimension for broadcasting
        return torch.exp(-(x - y)**2 / (2 * h**2)) / h

    def KR(x, X, Y):   # Kernel regression
        K = ker(x, X)
        Y = Y.unsqueeze(-2)
        y = torch.sum(Y * K, dim=-1)
        K_sum = torch.sum(K, dim=-1)
        return y / K_sum

    # weights normlization
    weights += 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
    dx = (bins[:,1] - bins[:,0]).unsqueeze(-1)
    bins_m = (bins[:,1:] + bins[:,:-1]) / 2
    KR_tensor = KR(bins, bins_m, pdf)
    integral_values_approx = (KR_tensor[:,:-1] + KR_tensor[:,1:]) * dx / 2
    cdf = torch.cumsum(integral_values_approx, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) / cdf.max()  # [N_rays, M+1] 

    if det: # True
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples] , device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)
    bins_g = torch.gather(bins, 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

# uniform
def sample_along_camera_ray(ray_o, ray_d, depth_range, N_samples, inv_uniform=False, det=False):
    """
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    """
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
    if inv_uniform:
        start = 1.0 / near_depth  # [N_rays,]
        step = (1.0 / far_depth - start) / (N_samples - 1)
        inv_z_vals = torch.stack(
            [start + i * step for i in range(N_samples)], dim=1
        )  # [N_rays, N_samples]
        z_vals = 1.0 / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples - 1)
        z_vals = torch.stack(
            [start + i * step for i in range(N_samples)], dim=1
        )  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o  # [N_rays, N_samples, 3]
    return pts, z_vals

def sample_fine_pts(inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals):
    if inv_uniform:
        inv_z_vals = 1.0 / z_vals
        inv_z_vals_mid = 0.5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])  # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
        inv_z_vals = sample_pdf_g(
            bins=torch.flip(inv_z_vals_mid, dims=[1]),
            weights=torch.flip(weights, dims=[1]),
            N_samples=N_importance,
            det=det,
        )  # [N_rays, N_importance]
        z_samples = 1.0 / inv_z_vals
    else:
        # take mid-points of depth samples
        z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
        z_samples = sample_pdf_g(
            bins=z_vals_mid, weights=weights, N_samples=N_importance, det=det
        )  # [N_rays, N_importance]

    z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance] 粗细阶段采样点合并
    z_vals, _ = torch.sort(z_vals, dim=-1)
    N_total_samples = N_samples + N_importance
 
    viewdirs = ray_batch["ray_d"].unsqueeze(1).repeat(1, N_total_samples, 1)
    ray_o = ray_batch["ray_o"].unsqueeze(1).repeat(1, N_total_samples, 1)
    pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]
    return pts, z_vals

def get_globfeat(ray_batch, img_feat, model):

    ray_d = ray_batch["ray_d"] #(N_rays,3)
    cam_e = ray_batch["camera"][:,-16:].squeeze(0).reshape(4,4)
    cam_pos = cam_e[:3,-1].unsqueeze(0).repeat(ray_d.shape[0],1) #(N_rays,3)

    latent_set = model.global_encoder(img_feat) #(n_view, 32, H/8, W/8)-->(n_view,h*w,32)
    glob_feat = model.global_decoder(latent_set, cam_pos, ray_d) #(N_rays,32)

    return glob_feat

def render_rays(
    ray_batch,
    model,
    featmaps,
    projector,
    N_samples,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    ret_alpha=False,
    single_net=True,
):
    """
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :param ret_alpha: if True, will return learned 'density' values inferred from the attention maps
    :param single_net: if True, will use single network, can be cued with both coarse and fine points
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    """

    glob_feat = get_globfeat(ray_batch, featmaps[0], model) #(N_rays,32)
    
    ret = {"outputs_coarse": None, "outputs_fine": None}
    ray_o, ray_d = ray_batch["ray_o"], ray_batch["ray_d"]

    # coarse
    # pts: [N_rays, N_samples, 3], z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(
        ray_o=ray_o,
        ray_d=ray_d,
        depth_range=ray_batch["depth_range"],
        N_samples=N_samples,
        inv_uniform=inv_uniform,
        det=det,
    )

    N_rays, N_samples = pts.shape[:2]
    rgb_feat, ray_diff, mask = projector.compute(
        pts,
        ray_batch["camera"],
        ray_batch["src_rgbs"],
        ray_batch["src_cameras"],
        featmaps=featmaps[1],
    )  # [N_rays, N_samples, N_views, x]

    rgb, coarse_feat = model.net_coarse(rgb_feat, ray_diff, mask, pts, ray_d, glob_feat, query_feat=None)
    if ret_alpha:
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        # depth_map = torch.sum(weights[:,1:] * z_vals, dim=-1)
        depth_map = None
    else:
        weights = None
        depth_map = None
    ret["outputs_coarse"] = {"rgb": rgb, "weights": weights, "depth": depth_map}

    # fine
    if N_importance > 0:
        weights = ret["outputs_coarse"]["weights"].clone().detach()  # [N_rays, N_samples]
        pts, z_vals = sample_fine_pts(
            True, N_importance, True, N_samples, ray_batch, weights, z_vals
        )
        rgb_feat_sampled, ray_diff, mask = projector.compute(
            pts,
            ray_batch["camera"],
            ray_batch["src_rgbs"],
            ray_batch["src_cameras"],
            featmaps=featmaps[1],
        )

        if single_net:
            rgb,_ = model.net_coarse(rgb_feat_sampled, ray_diff, mask, pts, ray_d, glob_feat, coarse_feat)
        else:
            rgb = model.net_fine(rgb_feat_sampled, glob_feat, ray_diff, mask, pts, ray_d)
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        # depth_map = torch.sum(weights[:,1:] * z_vals, dim=-1)
        depth_map = None
        
        rgb = rgb + ret["outputs_coarse"]["rgb"]
        ret["outputs_fine"] = {"rgb": rgb, "weights": weights, "depth": depth_map}

    return ret
