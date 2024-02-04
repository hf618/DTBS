import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform, color_jitter, gaussian_blur)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model1, ema_model2, model):  # ema_model两个
    for ema_param, param in zip(ema_model1.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False

    for ema_param, param in zip(ema_model2.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False

    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg1 = deepcopy(cfg['model'])  ####改
        self.ema_model1 = build_segmentor(ema_cfg1)

        ema_cfg2 = deepcopy(cfg['model'])  ####增加ema2
        self.ema_model2 = build_segmentor(ema_cfg2)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model1(self):
        return get_module(self.ema_model1)

    def get_ema_model2(self):  # 增加
        return get_module(self.ema_model2)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        # for ema_model2
        for param in self.get_ema_model1().parameters():
            param.detach_()
        # for ema_model2
        for param in self.get_ema_model2().parameters():
            param.detach_()

        mp = list(self.get_model().parameters())
        mcp1 = list(self.get_ema_model1().parameters())

        mcp2 = list(self.get_ema_model2().parameters())
        for i in range(0, len(mp)):
            if not mcp1[i].data.shape:  # scalar tensor
                mcp1[i].data = mp[i].data.clone()
            else:
                mcp1[i].data[:] = mp[i].data[:].clone()

            if not mcp2[i].data.shape:  # scalar tensor
                mcp2[i].data = mp[i].data.clone()
            else:
                mcp2[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        # for ema1
        for ema_param, param in zip(self.get_ema_model1().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

        # for ema2
        for ema_param, param in zip(self.get_ema_model2().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def _update_ema_TSF(self, iter,beta):
        w2 = beta
        w1 = 1-beta
        alpha_teacher2 = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param1 ,param2 in zip(self.get_model().parameters(),
                                    self.get_ema_model1().parameters(),
                                    self.get_ema_model2().parameters()):
            if (not param1.data.shape) and (not param2.data.shape):  # scalar tensor
                ema_param.data = \
                    alpha_teacher2 * ema_param.data + \
                    (1 - alpha_teacher2) * param1.data * w1+ (1 - alpha_teacher2) * param2.data * w2
            else:
                ema_param.data[:] = \
                    alpha_teacher2 * ema_param[:].data[:] + \
                    (1 - alpha_teacher2) * param1[:].data[:] * w1 + (1 - alpha_teacher2) * param2[:].data[:]* w2

    def _update_ema_TSF_E(self, iter,en_loss1,en_loss2):
        w1 = en_loss1/(en_loss2+en_loss1)
        w2 = en_loss2/(en_loss2+en_loss1)
        alpha_teacher2 = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param1 ,param2 in zip(self.get_model().parameters(),
                                    self.get_ema_model1().parameters(),
                                    self.get_ema_model2().parameters()):
            if (not param1.data.shape) and (not param2.data.shape):  # scalar tensor
                ema_param.data = \
                    alpha_teacher2 * ema_param.data + \
                    (1 - alpha_teacher2) * param1.data * w1+ (1 - alpha_teacher2) * param2.data * w2
            else:
                ema_param.data[:] = \
                    alpha_teacher2 * ema_param[:].data[:] + \
                    (1 - alpha_teacher2) * param1[:].data[:] * w1 + (1 - alpha_teacher2) * param2[:].data[:]* w2

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_day_img,
                      target_day_img_metas, target_night_img, target_night_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model1().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        for m in self.get_ema_model2().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits1 = self.get_ema_model1().encode_decode(
            target_day_img, target_day_img_metas)

        ema_softmax1 = torch.softmax(ema_logits1.detach(), dim=1)
        en_loss1 = entropy_loss(ema_softmax1)

        pseudo_prob1, pseudo_label1 = torch.max(ema_softmax1, dim=1)
        ps_large_p1 = pseudo_prob1.ge(self.pseudo_threshold).long() == 1
        ps_size1 = np.size(np.array(pseudo_label1.cpu()))
        pseudo_weight1 = torch.sum(ps_large_p1).item() / ps_size1
        pseudo_weight1 = pseudo_weight1 * torch.ones(
            pseudo_prob1.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight1[:, :self.psweight_ignore_top, :] = 0

        if self.psweight_ignore_bottom > 0:
            pseudo_weight1[:, -self.psweight_ignore_bottom:, :] = 0

        gt_pixel_weight1 = torch.ones((pseudo_weight1.shape), device=dev)  # 复制版本

        # Apply mixing
        mixed_img1, mixed_lbl1 = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]

            mixed_img1[i], mixed_lbl1[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_day_img[i])),  # 这里代表黄线
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label1[i])))

            _, pseudo_weight1[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight1[i], pseudo_weight1[i])))
        mixed_img1 = torch.cat(mixed_img1)
        mixed_lbl1 = torch.cat(mixed_lbl1)

        # Train on mixed images

        mix_losses1 = self.get_model().forward_train(
            mixed_img1, img_metas, mixed_lbl1, pseudo_weight1, return_feat=True)
        mix_losses1.pop('features')
        mix_losses1 = add_prefix(mix_losses1, 'mix1')  # mix day
        mix_loss1, mix_log_vars1 = self._parse_losses(mix_losses1)
        log_vars.update(mix_log_vars1)
        mix_loss1.backward()

        if self.local_iter > 0:
            self._update_ema2(self.local_iter)

        # for night
        ema_logits2 = self.get_ema_model2().encode_decode(
            target_night_img, target_night_img_metas)


        ema_softmax2 = torch.softmax(ema_logits2.detach(), dim=1)
        en_loss2 = entropy_loss(ema_softmax2)

        pseudo_prob2, pseudo_label2 = torch.max(ema_softmax2, dim=1)
        ps_large_p2 = pseudo_prob2.ge(self.pseudo_threshold).long() == 1
        ps_size2 = np.size(np.array(pseudo_label2.cpu()))
        pseudo_weight2 = torch.sum(ps_large_p2).item() / ps_size2
        pseudo_weight2 = pseudo_weight2 * torch.ones(
            pseudo_prob2.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight2[:, :self.psweight_ignore_top, :] = 0  # 复制版本

        if self.psweight_ignore_bottom > 0:
            pseudo_weight2[:, -self.psweight_ignore_bottom:, :] = 0

        gt_pixel_weight2 = torch.ones((pseudo_weight2.shape), device=dev)

        # Apply mixing

        mixed_img2, mixed_lbl2 = [None] * batch_size, [None] * batch_size
        # mix_masks = get_class_masks(gt_semantic_seg)


        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]

            mixed_img2[i], mixed_lbl2[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_night_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label2[i])))

            _, pseudo_weight2[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight2[i], pseudo_weight2[i])))
        mixed_img2 = torch.cat(mixed_img2)
        mixed_lbl2 = torch.cat(mixed_lbl2)

        # Train on mixed images


        mix_losses2 = self.get_model().forward_train(
            mixed_img2, img_metas, mixed_lbl2, pseudo_weight2, return_feat=True)
        mix_losses2.pop('features')
        mix_losses2 = add_prefix(mix_losses2, 'mix2')  # mix night
        mix_loss2, mix_log_vars2 = self._parse_losses(mix_losses2)
        log_vars.update(mix_log_vars2)
        mix_loss2.backward()


        # T-S feedback
        self._update_ema_TSF(self.local_iter,0.8)
        # T-S feedback(E)
        # self._update_ema_TSF_E(self.local_iter, en_loss1, en_loss2)


        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)

            vis_trg_img1 = torch.clamp(denorm(target_day_img, means, stds), 0, 1)
            vis_mixed_img1 = torch.clamp(denorm(mixed_img1, means, stds), 0, 1)


            vis_trg_img2 = torch.clamp(denorm(target_night_img, means, stds), 0, 1)
            vis_mixed_img2 = torch.clamp(denorm(mixed_img2, means, stds), 0, 1)


            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )


                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img1[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label1[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img1[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')

                subplotimg(
                    axs[1][3], mixed_lbl1[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight1[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}_day.png'))
                plt.close()


                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )


                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img2[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label2[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img2[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')

                subplotimg(
                    axs[1][3], mixed_lbl2[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight2[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}_night.png'))
                plt.close()




        self.local_iter += 1

        return log_vars
