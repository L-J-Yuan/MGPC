import torch
from extensions.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
import open3d as o3d

chamfer_dist = chamfer_3DDist()

def arcosh(x, eps=1e-5):
    # x = x.clamp(-1 + eps, 1 - eps)
    # x = x.clamp(1,)
    x = torch.clamp(x, min=1 + eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))


def CD(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def CD_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def HyperCD(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = arcosh(1+ 1 * d1)
    d2 = arcosh(1+ 1 * d2)

    return torch.mean(d1) + torch.mean(d2)


def CD_sqrt_partial(partial, pred):
    d1, d2, _, _ = chamfer_dist(partial, pred)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def F_score(pred, gt, th=0.01):
    def _get_open3d_ptcloud(tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = o3d.geometry.PointCloud()
        ptcloud.points = o3d.utility.Vector3dVector(tensor)

        return ptcloud

    b = pred.size(0)
    assert pred.size(0) == gt.size(0)
    if b != 1:
        f_score_list = []
        for idx in range(b):
            f_score_list.append(F_score(pred[idx:idx+1], gt[idx:idx+1], th))
        return sum(f_score_list)/len(f_score_list)
    else:
        pred = _get_open3d_ptcloud(pred)
        gt = _get_open3d_ptcloud(gt)

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))
        return 2 * recall * precision / (recall + precision) if recall + precision else 0


