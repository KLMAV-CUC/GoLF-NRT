import torch.nn as nn
from utils import img2mse


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log

def image_loss(model_output, gt, scalars_to_log):
    gt_rgb = gt['rgb']
    pred_rgb = model_output['rgb']
    loss = img2mse(pred_rgb, gt_rgb)
    return loss, scalars_to_log

class My_Loss(nn.Module):
    def __init__(self, l2_Lpips=0.02):
        super().__init__()
        self.l2_Lpips = l2_Lpips
        
        import lpips
        self.lpips_fn = lpips.LPIPS(net='vgg').cuda()
        # self.upsample = nn.Upsample((32,32), mode='bilinear')

    def forward(self, model_output, gt, scalars_to_log):
        loss_dic = {}
        loss_dic['img_loss'], scalars_to_log = image_loss(model_output, gt, scalars_to_log)

        gt_rgb = gt['rgb'].unsqueeze(0)
        pred_rgb = model_output['rgb'].unsqueeze(0)
        offset = 16
        # print(gt_rgb.shape)
        # print(pred_rgb.shape)
        gt_rgb = gt_rgb[:,:256,:].reshape(-1,offset,offset,3).permute(0,3,1,2)
        pred_rgb = pred_rgb[:,:256,:].reshape(-1,offset,offset,3).permute(0,3,1,2)

        lpips_loss = self.lpips_fn(gt_rgb, pred_rgb)
        loss_dic['lpips_loss'] = lpips_loss * self.l2_Lpips
        loss = loss_dic['img_loss'] + loss_dic['lpips_loss']

        return loss_dic, loss, scalars_to_log