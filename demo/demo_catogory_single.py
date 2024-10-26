# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import torch.multiprocessing as mp
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
from detectron2.engine.uda_instance_utils import visulize_color_instances # correct_label_by_CLIP, \
#correct_label_by_GT, remove_wrong_label_instance_by_GT,remove_empty_instance_by_GT, keep_stuff_label_instance_by_GT


from detectron2.data import MetadataCatalog
from cityscapesscripts.helpers.labels import name2label


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# constants
WINDOW_NAME = "mask2former demo"

EVAL=True
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
        # default=['/home/yguo/Documents/other/UDA4Inst/debug_cindy'],
        # default=['/datafast/120-1/Datasets/segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'],
        default=['datasets/urbansyn_total_label/img_urbansyn_instance_category_val.txt'],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='visual_instance/category_urbanysn_full_',
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
        # default=['MODEL.WEIGHTS','./output/smartmix/urbansyn_random_small_fix_20kbs3/model_best.pth'],
        # default=['MODEL.WEIGHTS','./output/smartmix/urbansyn_only_source_range_5_10/model_final.pth'],
        default=['MODEL.WEIGHTS','./output/category/urbansyn_full/model_final.pth'],
        
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

    if image_a.shape == image_b.shape:
        # 创建一个全零的图像，用于存储差异结果
        difference_image = np.zeros_like(image_a, dtype=np.uint8)

        # 定义白色像素值，表示无数据区域
        white = [255, 255, 255]

        # 创建掩码，标记哪些像素有数据（非白色）
        gt_has_data = np.any(image_a != white, axis=-1)
        pred_has_data = np.any(image_b != white, axis=-1)

        # 两个图像都在该像素位置有数据
        both_have_data = gt_has_data & pred_has_data

        # 检查在有数据的位置，两个图像的颜色是否不同
        colors_different = np.any(image_a != image_b, axis=-1)

        # 定义不同的掩码
        # 1. gt_color有，visual_pred_color没有：橙色
        mask_orange = gt_has_data & (~pred_has_data)

        # 2. gt_color没有，visual_pred_color有：灰色
        mask_gray = (~gt_has_data) & pred_has_data

        # 3. 都有数据，但颜色不同：蓝色
        mask_blue = both_have_data & colors_different

        # 4. 都有数据，且颜色相同：绿色
        mask_green = both_have_data & (~colors_different)

        # 5. 都没有数据：白色
        mask_white = (~gt_has_data) & (~pred_has_data)

        # 分配颜色（OpenCV使用BGR格式）
        difference_image[mask_orange] = [0, 165, 255]   # 橙色
        difference_image[mask_gray] = [128, 128, 128]   # 灰色
        difference_image[mask_blue] = [255, 0, 0]       # 蓝色
        difference_image[mask_green] = [0, 255, 0]      # 绿色
        difference_image[mask_white] = [255, 255, 255]  # 白色

        # 在error_map上添加图例
        error_map = difference_image.copy()
        error_map = add_legend(error_map)

        # 如果需要转换为RGB格式
        error_map_rgb = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

        return error_map_rgb
    else:
        print("输入的图像尺寸不一致。")
        return None

def add_legend(image):
    # 定义图例信息：颜色和对应的文本
    legend_info = [
        ([0, 165, 255], 'GT exit, No Pred'),      # 橙色
        ([128, 128, 128], 'No GT, Pred exit'),    # 灰色
        ([255, 0, 0], 'class error'),            # 蓝色
        ([0, 255, 0], 'correct'),            # 绿色
    ]

    # 图例起始位置
    start_x, start_y = 10, 10
    rect_size = 20  # 色块大小
    spacing = 5     # 间隔

    # 设置文字字体和大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    for i, (color, text) in enumerate(legend_info):
        # 计算位置
        y = start_y + i * (rect_size + spacing)
        # 绘制颜色方块
        cv2.rectangle(image, (start_x, y), (start_x + rect_size, y + rect_size), color, -1)
        # 添加文字（OpenCV的文字颜色是BGR）
        cv2.putText(image, text, (start_x + rect_size + spacing, y + rect_size - 5), 
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image


def get_error_map_bp(gt_color, visual_pred_color):
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

def cat_pred_gt(visual_pred_color, mask_vis_output, file_name):
    gt_path = file_name.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_color.png')
    train_id_path = file_name.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
    gt_img = cv2.imread(gt_path)
    rgb_img = cv2.imread(file_name)
    train_id_img = cv2.imread(train_id_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    human_cycle_vehicle_train_id = [11, 12, 13, 14, 15, 16, 17, 18]
    train_id_img_processed = np.where(np.isin(train_id_img, human_cycle_vehicle_train_id), train_id_img, 255)
    mask = train_id_img_processed==255
    mask_huam_cycle_behicle = ~mask * 1
    gt_img = gt_img * mask_huam_cycle_behicle
    black_pixels = np.all(gt_img == [0, 0, 0], axis=-1)
    gt_img[black_pixels] = [255, 255, 255]

    error_map = get_error_map(gt_img.astype("uint8"), visual_pred_color)

    h,w,c = rgb_img.shape
    black_bar_shape = (10, w, c)
    black_bar = np.zeros(black_bar_shape, dtype=np.uint8)
    rgb_blackbar = np.vstack((rgb_img, black_bar))
    rgb_mask = np.vstack((rgb_blackbar, mask_vis_output))
    rgb_mask_blackbar = np.vstack((rgb_mask, black_bar))
    rgb_mask_gt = np.vstack((rgb_mask_blackbar, gt_img))
    rgb_mask_gt_blackbar = np.vstack((rgb_mask_gt, black_bar))
    rgb_mask_gt_pred = np.vstack((rgb_mask_gt_blackbar,visual_pred_color))
    rgb_mask_gt_pred_blackbar = np.vstack((rgb_mask_gt_pred,black_bar))
    rgb_mask_gt_pred_errormap_blackbar = np.vstack((rgb_mask_gt_pred_blackbar,error_map))
    return rgb_mask_gt_pred_errormap_blackbar, error_map


def process_one(path, demo_human_cycle, _metadata, result_save_folder, visul_save_folder, other_map_save_folder):
    # use PIL, to be consistent with evaluation
    # print('-')
    path = str(path)
    # print('path of img is : ', path)
    img = read_image(path, format="BGR")
    out_filename = os.path.join(visul_save_folder, os.path.basename(path))
    predictions_fuse, visualized_output, visualizer = demo_human_cycle.run_on_image_for_instance(img)
    if VISUAL:
        visualized_output.save(out_filename.replace('.png', '_mask.png'))

    cpu_device = torch.device("cpu")
    instances = predictions_fuse['instances'].to(cpu_device)

    '''if the score(=class*mask)>0.5,I want to correct the score by CLIP, the class score will be updated by 
    CLIP output,and then update the score of the instance'''
    # instances = instances[instances.scores.cpu() > 0.5]
    # instances = correct_label_by_CLIP(instances, img, path, other_map_save_folder, debug_vis=VISUAL) #time consuming 2.8s
    # instances = remove_wrong_label_instance_by_GT(instances, path)
    # instances, correct_class_pair = correct_label_by_GT(instances, path)
    
    # instances = keep_stuff_label_instance_by_GT(instances, path)
    # instances = remove_empty_instance_by_GT(instances, path)
    
    # print('instances:', len(instances))
    # instances = instances[instances.scores.cpu() > 0.9]
    """ save instances for eval """
    num_instances = len(instances)
    file_name = path
    basename = os.path.splitext(os.path.basename(file_name))[0]
    pred_txt = os.path.join(result_save_folder, basename + "_pred.txt")
    if VISUAL and num_instances != 0:
        color_pseudo_instances = visulize_color_instances(instances)
        mask_img = cv2.imread(out_filename.replace('.png', '_mask.png'))
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        visual_pred = 255 * np.ones(img.shape, dtype=np.uint8)
        visual_pred_filename = os.path.join(visul_save_folder, basename + "_visual_pred.png")
        error_map_filename = os.path.join(visul_save_folder, basename + "_error_map.png")
        color_pseudo_instances_path = os.path.join(visul_save_folder, basename + "_instance.png")
        Image.fromarray(color_pseudo_instances).save(color_pseudo_instances_path)
    # if len(instances) != 0:
    with open(pred_txt, "w") as fout:
        for i in range(num_instances):
            pred_class = instances.pred_classes[i]
            classes = _metadata.thing_classes[pred_class]
            class_id = name2label[classes].id
            class_train_id = name2label[classes].trainId
            score = instances.scores[i]
            if VISUAL:
                # mask = instances.pred_masks[i].numpy().astype("uint8")
                mask = instances.pred_masks[i].bool()
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
            
        if VISUAL and num_instances != 0:
            visual_semantic_pred_color = process_train_id_to_color_img(visual_pred[:,:,0])
            # Image.fromarray(visual_pred_color).save(visual_pred_filename)
            final_img, error_map = cat_pred_gt(visual_semantic_pred_color, mask_img, file_name)
            Image.fromarray(final_img.astype("uint8")).save(visual_pred_filename)
            cv2.imwrite(out_filename.replace('.png', '_mask_text.png'), mask_img)
            Image.fromarray(error_map.astype("uint8")).save(error_map_filename)

def process_all(inputs, demo, result_save_folder, visul_save_folder, error_map_save_folder):
    _metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
    for path in tqdm.tqdm(inputs):
        process_one(path, demo, _metadata, result_save_folder, visul_save_folder, error_map_save_folder)


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
        model = model_weights[0]
        args.opts[1] = model
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)
        target = 'human_cycle_vehicle'
        folder = args.output
        result_save_folder = folder + cfg['MODEL']['WEIGHTS'].split('/')[-2] + '_instance_img'
        visul_save_folder = folder +  cfg['MODEL']['WEIGHTS'].split('/')[-2] + '_visul_img'
        other_map_save_folder = folder + cfg['MODEL']['WEIGHTS'].split('/')[-2] + '_other_map'
        print('result save path : ',result_save_folder)
        creat_empty_folder(result_save_folder)
        creat_empty_folder(visul_save_folder)
        creat_empty_folder(other_map_save_folder)
        args_input = args.input
        if len(args_input) == 1:
            if os.path.isdir(args_input[0]):
                inputs = sorted(Path(args_input[0]).glob('*/*.png'))
            elif os.path.isfile(args_input[0]):
                # inputs = glob.glob(os.path.expanduser(args_input[0]))
                # 读取文件中的每一行
                with open(args_input[0], 'r') as file:
                    lines = file.readlines()

                # 去掉每行末尾的换行符
                inputs = [line.strip() for line in lines]

        # process_all(inputs, demo, result_save_folder, visul_save_folder, error_map_save_folder)
        ''' multi-process '''

        _metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
        paramers = []
        for i in range(len(inputs)):
            paramers.append((inputs[i], demo, _metadata, result_save_folder, visul_save_folder, other_map_save_folder))
        # input :path, demo, _metadata, result_save_folder, visul_save_folder, error_map_save_folder
        pool = mp.Pool(processes=5)
        pool.starmap(process_one, paramers)
    if EVAL:
        # organise_evaluate_folder(result_save_folder)
        # organise_evaluate_folder(visul_save_folder)
        # result_save_folder='/home/yguo/Documents/other/UDA4Inst/visual_instance/urbanysn_uda_clip_urbansyn_random_small_fix_20kbs3_instance_img'
        os.environ['CITYSCAPES_RESULTS'] = result_save_folder
        # os.system('python /home/yguo/Documents/cityscapesScripts/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py')
        os.system('python /home/yguo/Documents/cityscapesScripts/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling_urbansyn.py')
