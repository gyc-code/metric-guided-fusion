# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
from pathlib import Path
import warnings
import os
import copy
from PIL import Image
import shutil

import cv2
import numpy as np
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from detectron2.structures import PolygonMasks, Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation.cityscapes_evaluation import process_train_id_to_color_img
from detectron2.engine.uda_instance_utils import visulize_color_instances


from detectron2.data import MetadataCatalog
from cityscapesscripts.helpers.labels import name2label


# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# constants
WINDOW_NAME = "mask2former demo"

EVAL = True
VISUAL = False
ONLY_VAL = False


"python demo.py --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="/home/yguo/Documents/other/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        default="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        # default=['/home/yguo/Documents/other/detectron2/demo/b.jpg'],
        # default=['/home/yguo/Documents/other/Mask2Former/danna_visual'],
        default=['datasets/synscapes/category_img_synscapes_instance_val.txt'],
        # default=['datasets/urbansyn_total_label/img_urbansyn_instance_small.txt'],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='visual_instance/category/category_synscapes_500_',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        # default=['MODEL.WEIGHTS','detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'],
        # default=['MODEL.WEIGHTS','./output/uda_synscapes_clean2_1024_bs3from_coco_huamn_cycle_t2s_s2t_motor_augum/model_best.pth ./output/uda_synscapes_clean2_1024_bs3from_coco_vehicle_t2s_s2t_train_augum/model_best.pth'],
        # default=['MODEL.WEIGHTS','./output/instan_seg/uda_urabn_human_cycle_1024_from_pre_coco_bs3_p0.9_t2s_s2t-motor-augu/model_best.pth ./output/instan_seg/uda_urabn_vehicle_1024_from_pre_coco_bs3_p0.9_t2s_s2t-train-source-augu/model_best.pth'],
        # default=['MODEL.WEIGHTS','./output/uda_synthia_human_cycle_1024_from_pre_coco_bs3_p0.9_0.25t2s_0.75s2t-motor-augu/model_best.pth ./output/uda_synthia_vehicle_1024_from_pre_coco_bs3_p0.9_0.25t2s_0.75s2t-bus-augu/model_best.pth'],
        default=['MODEL.WEIGHTS','./output/category/synscapes_human_cycle/model_final.pth ./output/category/synscapes_vehicle/model_final.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


def creat_empty_folder(folder_path): 
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

def organise_evaluate_folder(evaluate_folder):
    " for cityscape evaluate"
    city_names = ['frankfurt', 'lindau', 'munster']
    eval_path = Path(evaluate_folder)
    for city in city_names:
        eval_city_generate = eval_path.rglob(city + "*.png")
        eval_city_list = list(eval_city_generate)
        for i in eval_city_list:
            shutil.move(str(i), evaluate_folder + os.sep + city)

def get_error_map(gt_color, visual_pred_color):
    image_a = gt_color
    image_b = visual_pred_color

    if visual_pred_color.shape == gt_color.shape:

        # 创建一个全零的图像，用于存储差异结果
        difference_image = np.zeros_like(image_a, dtype=np.uint8)
        mask = np.all(image_a == [255, 255, 255], axis=-1) & np.all(image_b == [255, 255, 255], axis=-1)


        # 找到两张图像不同的地方
        difference_mask = np.any(image_a != image_b, axis=-1)

        # 在差异的位置使用红色表示
        difference_image[difference_mask] = [33, 33, 255]  # 红色 [B, G, R]

        # # 在相同的位置使用绿色表示
        same_mask = np.logical_not(difference_mask)
        difference_image[same_mask] = [108,238,108]  # 绿色 [B, G, R]
        difference_image[mask] = [255, 255, 255]

        error_map = difference_image
        error_map = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

    return error_map

def cat_pred_gt(visual_pred_color, mask_vis_output, file_name, GT='cityscapes'):
    if GT== 'cityscapes':
        gt_path = file_name.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_color.png')
        train_id_path = file_name.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
    elif GT == 'urbansyn':
        # '/home/yguo/Documents/other/detectron2/datasets/urbansyn/poblenou_terrain/rgb_translated_cityscapes/image_scene_013_beauty_0013.png'
        #/home/yguo/Documents/other/UDA4Inst/datasets/urbansyn_total_label/urbansyn_total_label/poblenou_image_scene_001_objectcolor_0001_gtFine_labelIds.png
        city_name = file_name.split('/')[-3]
        image_name = file_name.split('/')[-1].replace('_beauty_', '_objectcolor_').replace('.png', '_gtFine_labelIds.png')
        train_id_path = './datasets/urbansyn_total_label/urbansyn_total_label/' + city_name + image_name
        
        gt_path = None# no color gt for urbansyn
        train_id_path = file_name.replace('img_urbansyn_instance_category_val', 'label_urbansyn_instance_category_val').replace('.png', '_trainId.png')
        
    gt_img = cv2.imread(gt_path)
    # rgb_img = cv2.imread(file_name)
    train_id_img = cv2.imread(train_id_path)
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    human_cycle_vehicle_train_id = [11, 12, 13, 14, 15, 16, 17, 18]
    train_id_img_processed = np.where(np.isin(train_id_img, human_cycle_vehicle_train_id), train_id_img, 255)
    mask = train_id_img_processed==255
    mask_huam_cycle_behicle = ~mask * 1
    gt_img = gt_img * mask_huam_cycle_behicle
    black_pixels = np.all(gt_img == [0, 0, 0], axis=-1)
    gt_img[black_pixels] = [255, 255, 255]

    error_map = get_error_map(gt_img.astype("uint8"), visual_pred_color)

    # h,w,c = rgb_img.shape
    # black_bar_shape = (10, w, c)
    # black_bar = np.zeros(black_bar_shape, dtype=np.uint8)
    # rgb_blackbar = np.vstack((rgb_img, black_bar))
    # rgb_mask = np.vstack((rgb_blackbar, mask_vis_output))
    # rgb_mask_blackbar = np.vstack((rgb_mask, black_bar))
    # rgb_mask_gt = np.vstack((rgb_mask_blackbar, gt_img))
    # rgb_mask_gt_blackbar = np.vstack((rgb_mask_gt, black_bar))
    # rgb_mask_gt_pred = np.vstack((rgb_mask_gt_blackbar,visual_pred_color))
    # rgb_mask_gt_pred_blackbar = np.vstack((rgb_mask_gt_pred,black_bar))
    # rgb_mask_gt_pred_errormap_blackbar = np.vstack((rgb_mask_gt_pred_blackbar,error_map))
    rgb_mask_gt_pred_errormap_blackbar = None
    return rgb_mask_gt_pred_errormap_blackbar, error_map


def process_one(path, demo_human_cycle, demo_vehicle, _metadata, result_save_folder, visul_save_folder, error_map_save_folder, target):
    # use PIL, to be consistent with evaluation
    # print(path)
    path = str(path)
    # print('path of img is : ', path)
    img = read_image(path, format="BGR")
    img_copy = copy.deepcopy(img)
    out_filename = os.path.join(visul_save_folder, os.path.basename(path))
    predictions_human_cycle, visualized_output_human_cycle, visualizer = demo_human_cycle.run_on_image_for_instance(img)
    if VISUAL:
        visualized_output_human_cycle.save(out_filename.replace('.png', '_human_cyc.png'))
    predictions_vehicle, visualized_output_vehicle, visualizer = demo_vehicle.run_on_image_for_instance(img_copy)
    if VISUAL:
        visualized_output_vehicle.save(out_filename.replace('.png', '_vehicle.png'))
    # import time
    # s=time.time()
    predictions_fuse = Instances.cat([predictions_human_cycle['instances'], predictions_vehicle['instances']])
    # ss= time.time() - s
    # print('cat time is :',ss)
    cpu_device = torch.device("cpu")
    instances = predictions_fuse.to(cpu_device)
    # instances = instances[instances.scores.cpu() > 0.85]

    """ save instances for eval """
    predictions_human_cycle = demo_human_cycle.predictor(img)
    predictions_vehicle = demo_vehicle.predictor(img_copy)
    file_name = path
    basename = os.path.splitext(os.path.basename(file_name))[0]
    
    pred_txt = os.path.join(result_save_folder, basename + "_pred.txt")
    visual_pred = 255 * np.ones(img.shape, dtype=np.uint8)
    visual_pred_filename = os.path.join(visul_save_folder, basename + "_visual_pred.png")
    error_map_filename = os.path.join(visul_save_folder, basename + "_error_map.png")
    if VISUAL:
        mask_vis_output = visualizer.draw_instance_predictions(predictions=instances)
        color_pseudo_instances_path = os.path.join(visul_save_folder, basename + "_instance.png")
        color_pseudo_instances = visulize_color_instances(instances)
        Image.fromarray(color_pseudo_instances).save(color_pseudo_instances_path)
        mask_vis_output.save(out_filename.replace('.png', '_mask.png'))
        mask_img = cv2.imread(out_filename.replace('.png', '_mask.png'))
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

    num_instances = len(instances)
    with open(pred_txt, "w") as fout:
        for i in range(num_instances):
            pred_class = instances.pred_classes[i]
            classes = _metadata.thing_classes[pred_class]
            class_id = name2label[classes].id
            class_train_id = name2label[classes].trainId
            score = instances.scores[i]
            # mask = instances.pred_masks[i].numpy().astype("uint8")
            mask = instances.pred_masks[i].bool()
            if VISUAL:
                true_positions = torch.where(mask)
                if true_positions[0].numel() > 0:
                    mid_0 = int(0.5*(true_positions[0][-1].item() - true_positions[0][0].item())) + true_positions[0][0].item()
                    mid_1 = int(0.5*(true_positions[1][-1].item() - true_positions[1][0].item())) + true_positions[1][0].item()
                    if mid_1 > 1995:
                        mid_1 = 1995
                    text_position = (mid_1, mid_0)
                    cv2.putText(mask_img, classes + str(round(score.item(), 2)), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            visual_pred[mask] = class_train_id
            png_filename = os.path.join(
                result_save_folder, basename + "_{}_{}.png".format(i, classes)
            )
            Image.fromarray(instances.pred_masks[i].numpy().astype("uint8") * 255).save(png_filename)
            fout.write("{} {} {}\n".format(os.path.basename(png_filename), class_id, score))
        if VISUAL:
            visual_semantic_pred_color = process_train_id_to_color_img(visual_pred[:,:,0])
            # Image.fromarray(visual_pred_color).save(visual_pred_filename)
            final_img, error_map = cat_pred_gt(visual_semantic_pred_color, mask_img, file_name, GT='urbansyn')
            # Image.fromarray(final_img.astype("uint8")).save(visual_pred_filename)
            cv2.imwrite(out_filename.replace('.png', '_mask_text.png'), mask_img)
            Image.fromarray(error_map.astype("uint8")).save(error_map_filename)


def process_all(inputs, demo_human_cycle, demo_vehicle, result_save_folder, visul_save_folder, error_map_save_folder, target):
    _metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
    for path in tqdm.tqdm(inputs):
        process_one(path, demo_human_cycle, demo_vehicle, _metadata, result_save_folder, visul_save_folder, error_map_save_folder, target)


if __name__ == "__main__":
    if not ONLY_VAL:
        mp.set_start_method("spawn", force=True)
        args = get_parser().parse_args()
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(args))
        '''  input multi model, seperate by ' ', run loop'''
        args_copy = copy.deepcopy(args)
        model_weights = args_copy.opts[1].split(' ')
        model_human_cycle = model_weights[0]
        model_vehicle = model_weights[1]

        args.opts[1] = model_human_cycle
        cfg = setup_cfg(args)
        demo_human_cycle = VisualizationDemo(cfg)

        args.opts[1] = model_vehicle
        cfg = setup_cfg(args)
        demo_vehicle = VisualizationDemo(cfg)

        target = 'human_cycle_vehicle'
        folder = args.output

        result_save_folder = folder + cfg['MODEL']['WEIGHTS'].split('/')[-1] + '_instance_img'
        visul_save_folder = folder +  cfg['MODEL']['WEIGHTS'].split('/')[-1] + '_visul_img'
        other_map_save_folder = folder + cfg['MODEL']['WEIGHTS'].split('/')[-1] + '_other_map'
        creat_empty_folder(result_save_folder)
        creat_empty_folder(visul_save_folder)
        creat_empty_folder(other_map_save_folder)

        args_input = args.input
        if len(args_input) == 1:
            if os.path.isdir(args_input[0]):
                inputs = sorted(Path(args_input[0]).glob('*/*.png'))
                # final_result = '/home/yguo/Documents/other/detectron2/final_cityscapes_val_result'

            elif os.path.isfile(args_input[0]):
                # inputs = glob.glob(os.path.expanduser(args_input[0]))
                # 读取文件中的每一行
                with open(args_input[0], 'r') as file:
                    lines = file.readlines()

                # 去掉每行末尾的换行符
                inputs = [line.strip() for line in lines]
                
        # process_all(inputs, demo_human_cycle, demo_vehicle, result_save_folder, visul_save_folder, other_map_save_folder, target)
        
        ''' multi-process '''
        _metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
        paramers = []
        for i in range(len(inputs)):
            paramers.append((inputs[i], demo_human_cycle, demo_vehicle, _metadata, result_save_folder, visul_save_folder, other_map_save_folder, target))
        # input :path, demo, _metadata, result_save_folder, visul_save_folder, error_map_save_folder
        pool = mp.Pool(processes=2)
        pool.starmap(process_one, paramers)
    
    if EVAL:
        # organise_evaluate_folder(result_save_folder)
        # organise_evaluate_folder(visul_save_folder)
        os.environ['CITYSCAPES_RESULTS'] = result_save_folder
        # os.environ['CITYSCAPES_RESULTS'] = 'visual_instance/category/category_urbansyn_1model_final.pth_instance_img'
        # os.system('python /home/yguo/Documents/cityscapesScripts/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py')
        os.system('python /home/yguo/Documents/cityscapesScripts/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling_urbansyn.py')

