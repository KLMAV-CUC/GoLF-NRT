import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
import torch.utils.tensorboard as tb

from torch.utils.data import DataLoader

from golf.data_loaders import dataset_dict
from golf.render_ray import render_rays
from golf.render_image import render_single_image
from golf.model import GoLFModel
from golf.sample_ray import RaySamplerSingleImage
from golf.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, img2psnr
import config
import torch.distributed as dist
from golf.projection import Projector
from golf.data_loaders.create_training_dataset import create_training_dataset
import imageio
from os import path


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):
    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)
    ##############################################################

    train_dataset, train_sampler = create_training_dataset(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        worker_init_fn=lambda _: np.random.seed(),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )
    val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(val_loader)
    ##############################################################

    model = GoLFModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    projector = Projector(device=device)
    criterion = Criterion()
    ##############################################################
    scalars_to_log = {}

    os.makedirs(args.rootdir, exist_ok=True)
    shutil.rmtree(path.join(args.rootdir, 'log/train'), ignore_errors=True)
    logger = tb.SummaryWriter(path.join(args.rootdir, 'log/train'), flush_secs=1)
    ##############################################################

    global_step = model.start_step + 1
    epoch = 0
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()
            if args.distributed:
                train_sampler.set_epoch(epoch)
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(
                1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
            ) 
            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=args.sample_mode,
                center_ratio=args.center_ratio,
            )
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            ret = render_rays(
                ray_batch=ray_batch,
                model=model,
                projector=projector,
                featmaps=featmaps,
                N_samples=args.N_samples,
                inv_uniform=args.inv_uniform,
                N_importance=args.N_importance,
                det=args.det,
                white_bkgd=args.white_bkgd,
                ret_alpha=args.N_importance > 0,
                single_net=args.single_net,
            ) 

            model.optimizer.zero_grad()
            if ret["outputs_fine"] is not None:
                fine_loss, scalars_to_log = criterion(ret["outputs_fine"], ray_batch, scalars_to_log)
                loss = fine_loss
            loss.backward()
            scalars_to_log["loss"] = loss.item()

            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    if ret["outputs_fine"] is not None:
                        mse_error = img2mse(ret["outputs_fine"]["rgb"], ray_batch["rgb"]).item()
                        scalars_to_log["train/fine-psnr-training-batch"] = mse2psnr(mse_error)


                    logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                    print(logstr)
                    print("each iter time {:.05f} seconds".format(dt))

                if global_step % 1000 == 0:
                    logger.add_scalar('train/loss_mse', loss.item(), global_step=global_step)

                if global_step % args.i_weights == 0:
                    print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                    fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0:
                    print("Logging a random validation view...")
                    psnr_scores = []
                    index = 1
                    while True:
                        try:
                            val_data = next(val_loader_iterator)
                        except:
                            val_loader_iterator = iter(val_loader)
                            break
                        tmp_ray_sampler = RaySamplerSingleImage(
                            val_data, device, render_stride=args.render_stride
                        )
                        H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                        gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                        psnr_curr_img = log_view(
                            global_step,
                            index,
                            args,
                            model,
                            tmp_ray_sampler,
                            projector,
                            gt_img,
                            render_stride=args.render_stride,
                            prefix="val/",
                            out_folder=out_folder,
                            ret_alpha=args.N_importance > 0,
                            single_net=args.single_net,
                        )
                        psnr_scores.append(psnr_curr_img)
                        index += 1
                    psnr_mean = np.mean(psnr_scores)
                    print("Average PSNR: ", psnr_mean)
                    torch.cuda.empty_cache()
                    logger.add_scalar('val/psnr', psnr_mean, global_step=global_step)
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


@torch.no_grad()
def log_view(
    global_step,
    index,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    single_net=True,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            ret_alpha=ret_alpha,
            single_net=single_net,
        )

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1])
    rgb_im = torch.zeros(3, h_max, 2 * w_max)
    rgb_im[:, : rgb_gt.shape[-2], : rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2], w_max : w_max + rgb_pred.shape[-1]] = rgb_pred
    depth_im = None

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    rgb_im = np.uint8(np.clip((rgb_im * 255.0), 0, 255))
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_{:01d}.png".format(global_step,index))
    imageio.imwrite(filename, rgb_im)
    if depth_im is not None:
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, prefix[:-1] + "depth_{:03d}.png".format(global_step))
        imageio.imwrite(filename, depth_im)

    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None else ret["outputs_coarse"]["rgb"]
    )
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print(prefix + "psnr_image: ", psnr_curr_img)
    model.switch_to_train()
    return psnr_curr_img


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(args.local_rank)

    train(args)
