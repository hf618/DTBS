# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction
# 丁哥改编for循环版本

import os
from argparse import ArgumentParser

import mmcv
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('imgs', help='Image folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    # test multiple images
    directory = args.imgs
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
        result = inference_segmentor(model, f)
    
        # show the results
        file, extension = os.path.splitext(f)
        #pred_file = f'{file}_4pred{extension}'
        pred_file = f'{file}*{extension}'
        assert pred_file != f
        model.show_result(
            f,
            result,
            palette=get_palette(args.palette),
            out_file=pred_file,
            show=False,
            opacity=args.opacity)
        print('Save prediction to', pred_file)
    
    
  


if __name__ == '__main__':
    main()
