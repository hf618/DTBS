import glob 
import os
from argparse import ArgumentParser
import numpy as np
from PIL import Image
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
    
    save_path = 'demo/save/'
    for city in os.listdir(directory):
        f_origin = os.path.join(directory, city)
        f_save = os.path.join(save_path,city)
        print(f_save)
   
        
        if not os.path.exists(f_save): #不存在就建立dic
            os.mkdir(f_save)
          
        for image in os.listdir(f_origin):
            if not image.endswith('png'):
                continue

            f = os.path.join(f_origin, image)
            
            # checking if it is a file
            if os.path.isfile(f):
                print(f)
            result = inference_segmentor(model, f)


            file, extension = os.path.splitext(image)
            file = file.replace('_rgb_anon', '*')
            pred_file = f'{file}{extension}'
            s = os.path.join(f_save, pred_file)
            #print(pred_file)
            re = result[0].astype(np.uint8)
            im = Image.fromarray(re)
            im.save(s)
            print('Save prediction to', s)
        '''
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
        '''
if __name__ == '__main__':
    main()
