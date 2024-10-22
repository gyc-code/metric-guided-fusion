import torch
import cv2
import numpy as np
import torch.nn.functional as F
import copy
import random
import matplotlib.pyplot as plt


from skimage import measure
from PIL import Image
import time
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from detectron2.structures.masks import polygons_to_bitmask

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage
from detectron2.structures import PolygonMasks, Instances
from collections import Counter

__all__ = [
"instance_poly2color_semantic",
"generate_class_mask",
"source_target_mix",
"instance_poly2color_semantic",
"apply_black_instance_on_target",
"source_data_augmentation",
"fillhole",
"transform_instance_annotations",
"filter_pseudo_instance",
]

DEBUG_IMG_FLAG = False
VISUALIZE_POLYGON = False
visual_iter = 500
Target_coefficients = None
Source_coefficients = None

# RARE_CLASS_NAMES = [3, 4, 5, 6, 7] # bus is 4, train is 5,  motor is 6, bike is 7
# RARE_CLASS_NAMES = [5, 6] # 3 for truck ,bus is 4, train is 5,  motor is 6, bike is 7
RARE_CLASS_NAMES = [] # close rare balance for ablation  TODO : TODO SHIFT FOR THIS
# RARE_CLASS_NAMES = [4, 6] # for synthia, bus is 4,  motor is 6

def translated_obj_mask(obj_mask,image, dx=50,dy=50):
    ''' dx control col, dy control row,dy > 0, move down, dx > 0, move right'''
    if dx==0 and dy==0:
        return obj_mask, image
    # 获取mask的形状
    rows, cols = obj_mask.shape
    # 创建平移后的mask
    translated_mask = torch.zeros_like(obj_mask, dtype=torch.bool)
    translated_img = copy.deepcopy(image)
    # 确定新的位置
    x_start = max(dx, 0)
    x_end = min(cols, cols + dx)
    y_start = max(dy, 0)
    y_end = min(rows, rows + dy)
    
    orig_x_start = max(-dx, 0)
    orig_x_end = min(cols, cols - dx)
    orig_y_start = max(-dy, 0)
    orig_y_end = min(rows, rows - dy)
    
    translated_mask[y_start:y_end, x_start:x_end] = obj_mask[orig_y_start:orig_y_end, orig_x_start:orig_x_end]
    for c in range(3):
        translated_img[c, y_start:y_end, x_start:x_end] = image[c, orig_y_start:orig_y_end, orig_x_start:orig_x_end]
    
    # cv2.imwrite('before.png',(obj_mask*255).numpy())
    # cv2.imwrite('after.png',(translated_mask*255).numpy())
 
    # cv2.imwrite('before_img.png',(image).permute(1,2,0).numpy())
    # cv2.imwrite('after_img.png',(translated_img).permute(1,2,0).numpy())

    return translated_mask, translated_img

def get_cityscapes_labels():
    return [
        # [128, 64, 128],0
        # [244, 35, 232],1
        # [70, 70, 70],2
        # [102, 102, 156],3
        # [190, 153, 153],4
        # [153, 153, 153],5
        # [250, 170, 30],6
        # [220, 220, 0],7
        # [107, 142, 35],8
        # [152, 251, 152],9
        # [0, 130, 180],10
        (220, 20, 60),# 11
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)]

def remove_occlussion(base_instances, pasted_instances):
    ''' remove the parts of base_instances coverd by pasted_instances '''
    # creat a mask for all instances for occlusion
    all_pasted_masks = pasted_instances.gt_masks.any(dim=0)
    # use ~ to get the un coverd region
    not_pasted_mask = ~all_pasted_masks
    # only keep uncovered region
    base_instances.gt_masks &= not_pasted_mask

    # slow version
    # paste_inst_num = pasted_instances.gt_masks.shape[0]
    # for i in range(paste_inst_num):
    #     not_past_inst = (~pasted_instances.gt_masks[i].unsqueeze(0)).int()
    #     base_instances.gt_masks = (not_past_inst * base_instances.gt_masks.int()).bool()
    # print(torch.equal(a, base_instances.gt_masks )) 

def remove_ego_car_logo(pseudo_instance, template):
    ''' design for cityscapes, crop 1024*1024, remove pseudo label which is ego car head and logo'''
    if len(pseudo_instance._fields) == 0:
        return
    white_mask = template >= 20
    white_mask = (white_mask*1)[0, :, :]
    pred_masks = pseudo_instance.pred_masks
    if len(pred_masks) == 0:
        return
    ''' process every mask , if one mask is covered totally by template, remove it and its score and label'''
    indices_to_remove = []
    for i, mask in enumerate(pred_masks):
        mask_area = mask.sum().item()
        if mask_area == 0:
            continue
        ''' mask:1 is area of instance, white_mask 1 is not ego car log, 0 is logo, 
        if mask is in white_mask, multipy makes it 0.'''
        template_apply_mask = mask * white_mask.cuda()
        # cv2.imwrite(str(i) + '_.png', template_apply_mask.cpu().numpy())
        # rows, cols = np.where(mask.cpu().numpy() > 0 )
        # center_row = 0.5*(rows.max() - rows.min()) + rows.min()
        template_apply_mask_area = template_apply_mask.sum().item()
        if (template_apply_mask_area/mask_area) < 0.2:
            indices_to_remove.append(i)
    new_pseudo_instance = Instances((pseudo_instance.image_size[0], pseudo_instance.image_size[1]))
    for i in range(len(pseudo_instance)):
        if i not in indices_to_remove:
            if len(new_pseudo_instance._fields) == 0:
                new_pseudo_instance = Instances.cat([pseudo_instance[i], pseudo_instance[i]])
            else:
                new_pseudo_instance = Instances.cat([new_pseudo_instance, pseudo_instance[i]])
    del pseudo_instance
    return new_pseudo_instance[1:]

def break_source_target_match(data):
    ''' break source and target match'''
    pass
    # print('================ data ', data[0]['source']['file_name'], '/n', data[0]['target']['file_name'])
    # print('                 data ', data[1]['source']['file_name'], '/n', data[1]['target']['file_name'])
    # print('                 data ', data[2]['source']['file_name'], '/n', data[2]['target']['file_name'])

def rare_class_balance(rare_class_samples, img_to_paste, instance_to_add):
    ''' 
    rare_class_samples : a list , element is map, {'img': img, 'instance': instance}
    img_to_paste: to paste rare instance to the img
    instance_to_add :  to add rare instance to the instance
    '''
    pick_num = int(len(rare_class_samples)/2) if len(rare_class_samples) > 1 else 1
    pick_samples = random.sample(rare_class_samples, pick_num)
    for i, sample in enumerate(pick_samples):
        img = sample['img']
        instance = sample['instance']
        mask = instance.gt_masks[0]
        for c in range(3):
            img_to_paste[c,:][mask] = img[c,:][mask]
        remove_occlussion(instance_to_add, instance)
        instance_to_add = Instances.cat([instance_to_add, instance])
    if len(rare_class_samples) > 10:# control the canditate number 
        del rare_class_samples[:4]
    return instance_to_add, rare_class_samples

def visulize_color_instances(instances):
    height = instances._image_size[0]
    width = instances._image_size[1]
    color_instances = np.zeros((height, width,3), dtype=np.uint8)
    try:
        instance_mask = instances.gt_masks
    except:
        instance_mask = (instances.pred_masks).cpu().bool()
    for i in range(len(instances)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_instances[instance_mask[i,:,:]] = [r,g,b]

    return color_instances

def polyfit(pseudo_instances):
    center_row_list = []
    height_list = []
    for i, pred_mask in enumerate(pseudo_instances.pred_masks):
        foreground_indices = pred_mask.nonzero(as_tuple=False)
        if foreground_indices.size(0) == 0:
            continue
            # raise ValueError("The pred_mask contains no foreground pixels.")
        # 计算中心坐标
        center_row = foreground_indices.float().mean(dim=0)[0].item()
        # print('center : ', foreground_indices.float().mean(dim=0))
        # 计算前景高度
        min_row = foreground_indices[:, 0].min().item()
        max_row = foreground_indices[:, 0].max().item()
        height = max_row - min_row + 1
        center_row_list.append(center_row)
        height_list.append(height)

    if len(center_row_list) > 2:
        # 示例数据：目标高度数组和最低点数组
        height_list = np.array(height_list)
        center_row_list = np.array(center_row_list)
        # 使用NumPy的polyfit函数拟合二次多项式
        Target_coefficients = np.polyfit(height_list, center_row_list, 1)
        ########  to show ###########################
        # # 创建一个多项式对象
        # polynomial = np.poly1d(Target_coefficients)
        # # 生成预测值
        # heights_fit = np.linspace(min(height_list), max(height_list), 100)
        # center_row_list = polynomial(heights_fit)
        # # 打印拟合的多项式系数
        # print(f"Fitted polynomial coefficients: {Target_coefficients}")
        # # 可视化
        # plt.figure(figsize=(10, 6))
        # plt.scatter(heights_fit, center_row_list, color='red', label='Data points')
        # plt.plot(heights_fit, center_row_list, color='blue', label='Fitted quadratic polynomial')
        # plt.xlabel('Height')
        # plt.ylabel('Lowest Point')
        # plt.title('Quadratic Polynomial Fitting')
        # plt.legend()
        # plt.grid(True)
        # # 保存图片
        # plt.savefig('quadratic_polynomial_fitting.png')
        ########  to show ###########################
    else:
        print('use last target_coefficients')
        # 计算前景高度
    min_row = foreground_indices[:, 0].min().item()
    max_row = foreground_indices[:, 0].max().item()
    height = max_row - min_row + 1

    # if Target_coefficients is not None:
    #     foreground_indices = obj_mask.nonzero(as_tuple=False)
    #     # 计算中心坐标
    #     center_row = int(foreground_indices.float().mean(dim=0)[0].item())
    #     # 创建一个多项式对象
    #     polynomial = np.poly1d(Target_coefficients)
    #     # 生成预测值
    #     # heights_fit = np.linspace(min(height_list), max(height_list), 100)
    #     center_row_fit = int(polynomial(height))
    #     # x_shift = random.randint(-150, 150)
    #     x_shift = 0
    #     y_shift = center_row_fit - center_row
    #     # print(center_row_fit, center_row, y_shift)
    # else:
    #     ''' shift obj_mask'''
    #     x_shift = random.randint(-150, 150)
    #     if height < 100: # TODO : CHANEG TO height
    #         y_shift = random.randint(-100, -50)
    #     else:
    #         y_shift = random.randint(0, 100)
    #     # x_shift = 0
    #     # y_shift = 0

def get_object_shift_by_depth_map(obj_mask, obj_depth_map, depth_map_to_paste):
    ''' depth of object gotten from obj_depth_map is the source depth, 
    and find this depth in depth_map_to_paste to know where to paste.
    the row to paste is important to know and the col can be random shift
    '''
    depths_array = obj_depth_map[obj_mask]
    counts = np.bincount(depths_array)
    obj_depth = np.argmax(counts)
    region_in_paste_img = depth_map_to_paste == obj_depth
    foreground_coords = np.column_stack(np.where(region_in_paste_img))
    if foreground_coords.size == 0:
        print('no this depth in image to paste')
        return 0, 0
    depth_center_y_to_paste, depth_center_x_to_paste = foreground_coords.mean(axis=0) ## TODO BUG . need depth of road surface  use person person
    obj_foreground_coords = np.column_stack(np.where(obj_mask))
    obj_center_y, obj_center_x = obj_foreground_coords.mean(axis=0)
    # cv2.imwrite('obj_area.png',((obj_mask*1)*255).numpy())
    # cv2.imwrite('obj_target_depth_' + str(obj_depth) + '.png',((region_in_paste_img*1)*255))
    return int(depth_center_x_to_paste - obj_center_x), int(depth_center_y_to_paste - obj_center_y)

def source_instance_paste_to_target_mix(one_data, local_iter, folder_name, source_rare_class_samples):
    global Target_coefficients
    depth_map_source = one_data['source']['depth'].astype(int)
    depth_map_target = one_data['target']['depth'].astype(int)
    gt_instance = one_data['source']['instances']
    gt_classes = gt_instance.gt_classes
    # gt_polygons = gt_syn.gt_masks
    gt_masks = gt_instance.gt_masks
    source_img = one_data['source']['image']
    _, hs, ws = source_img.shape

    target_img = one_data['target']['image']
    pseudo_instances = one_data['target']['instances']
    # pseudo_instances = pseudo_label['instances']
    file_id = one_data['target']['image_id'].split('.')[0]

    if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
        target_img_vis = target_img.cpu().permute(1,2,0).numpy()
        target_img_vis = cv2.cvtColor(target_img_vis,cv2.COLOR_BGR2RGB)
        cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_target_ori.jpg', target_img_vis)
        instances_img = 255 * np.ones(target_img.shape, dtype=np.uint8)
        gt_color_instances = visulize_color_instances(gt_instance)

    gt_instance_select = Instances((hs, ws))
    THIS_FRAME_HAS_RARE_CLASSES = False

    for i, obj_mask in enumerate(gt_masks):
        instance_size = (obj_mask*1).sum().item()
        if instance_size == 0:
            continue 
        # x_shift, y_shift = get_object_shift_by_depth_map(obj_mask, depth_map_source, depth_map_target)
        ''' shift obj_mask'''
        # x_shift = random.randint(-250, 250)
        # if instance_size < 5000: 
        #     y_shift = random.randint(-100, -50)
        # else:
        #     y_shift = random.randint(0, 100)
        x_shift, y_shift = 0, 0

        shift_obj_mask, shift_source_image = translated_obj_mask(obj_mask,source_img, dx=x_shift,dy=y_shift)        
        ins = Instances((hs, ws))
        ins.gt_classes = gt_classes[i].view(1)
        ins.gt_masks = shift_obj_mask.view(1, hs, ws)
        ''' gather all big source instances'''
        if len(gt_instance_select._fields) == 0:
            gt_instance_select = Instances.cat([ins, ins]) # first one is redandence
        else:
            gt_instance_select = Instances.cat([gt_instance_select, ins])
        ''' mix the image'''
        if i == 0:
            source_img = torch.from_numpy(cv2.GaussianBlur(source_img.permute(1,2,0).to(torch.uint8).numpy(), (5, 5),0)).permute(2,0,1)
        for c in range(3):
            # target_img[c,:][shift_obj_mask] = source_img[c,:][obj_mask]
            target_img[c,:][shift_obj_mask] = shift_source_image[c,:][shift_obj_mask]
            if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
                # instances_img[c,:][shift_obj_mask] = source_img[c,:][obj_mask]
                instances_img[c,:][shift_obj_mask] = shift_source_image[c,:][shift_obj_mask]

        # for rare class balance
        if gt_classes[i].item() in RARE_CLASS_NAMES:
            source_rare_class_samples.append({'img':source_img, 'instance':ins})
            THIS_FRAME_HAS_RARE_CLASSES = True

    if len(gt_instance_select._fields) != 0: # if gt_instance_selecthas nothing, no labels mix
        # do class balance
        if not THIS_FRAME_HAS_RARE_CLASSES and len(source_rare_class_samples):
            # print('rare class sample : ', len(source_rare_class_samples))
            gt_instance_select, source_rare_class_samples = rare_class_balance(source_rare_class_samples, target_img, gt_instance_select)
    
        pseudo_instances.gt_masks = pseudo_instances.pred_masks.bool().cpu()
        pseudo_instances.gt_classes = pseudo_instances.pred_classes.cpu()
        del pseudo_instances._fields['pred_masks']
        del pseudo_instances._fields['pred_classes']
        del pseudo_instances._fields['pred_boxes']
        del pseudo_instances._fields['scores']
        remove_occlussion(pseudo_instances, gt_instance_select[1:]) # modify pseudo_instances, remove parts of coverd by gt_instance_select
        one_data['source']['instances'] = Instances.cat([pseudo_instances, gt_instance_select[1:]])
        one_data['source']['image'] = target_img
        if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
            color_instances = visulize_color_instances(one_data['source']['instances'])
            target_img_vis = target_img.cpu().permute(1,2,0).numpy()
            source_img_vis = source_img.cpu().permute(1,2,0).numpy()
            target_img_vis = cv2.cvtColor(target_img_vis,cv2.COLOR_BGR2RGB)
            source_img_vis = cv2.cvtColor(source_img_vis,cv2.COLOR_BGR2RGB)
            target_img_vis_cp = copy.deepcopy(target_img_vis)
            if VISUALIZE_POLYGON:   
                color_map = get_cityscapes_labels()
                for i in range(one_data['source']['instances'].__len__()):
                    mask = one_data['source']['instances'].gt_masks[i].to(torch.uint8).numpy()
                    class_id = one_data['source']['instances'].gt_classes[i].item()
                    # classes = self._metadata.thing_classes[pred_class]
                    #     class_id = name2label[classes].id
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        cv2.drawContours(target_img_vis_cp, [contour], -1, (color_map[class_id][2],color_map[class_id][1], color_map[class_id][0]), 2)  # 绿色轮廓
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_s2t_mix.jpg', target_img_vis_cp)
            instances_img = instances_img.transpose((1,2,0))
            instances_img = cv2.cvtColor(instances_img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_s2t_instance.jpg', instances_img)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_s2t_color_instance.jpg', color_instances)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_source_gt_color_instance.jpg', gt_color_instances)

            # Image.fromarray(target_img_vis_cp).save(folder_name + file_id + '_' + str(local_iter) + '_s2t_mix---.jpg')
            # Image.fromarray(source_img_vis).save(folder_name + file_id + '_' + str(local_iter) + '_gt_img.jpg')
            city = one_data['source']['file_name'].split('/')[8]
            s_img_name = one_data['source']['file_name'].split('/')[-1]
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_gt_img_'  + city + '_' + s_img_name, source_img_vis)
            del target_img_vis_cp
            # instance_poly2color_semantic(pseudo_instances, hs, ws, folder_name, file_id, local_iter, flag='mix')
    return one_data, source_rare_class_samples


def target_instance_paste_to_source_mix(one_data, local_iter, folder_name, target_rare_class_samples=[]):
    gt_instance = one_data['source']['instances']
    gt_classes = gt_instance.gt_classes
    depth_map_source = one_data['source']['depth'].astype(int)
    depth_map_target = one_data['target']['depth'].astype(int)
    # gt_polygons = gt_syn.gt_masks
    gt_masks = gt_instance.gt_masks
    source_img = one_data['source']['image']
    _, hs, ws = source_img.shape

    target_img = one_data['target']['image']
    pseudo_instances = one_data['target']['instances']
    # pseudo_instances = pseudo_label['instances']
    pred_masks = pseudo_instances.pred_masks
    pred_classes = pseudo_instances.pred_classes
    file_id = one_data['target']['image_id'].split('.')[0]

    # if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
    #     cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_target_img_black.jpg', target_img.permute(1,2,0).numpy())
    #     instance_poly2color_semantic(gt_instance, hs, ws, folder_name, file_id, local_iter)
    #     instance_poly2color_semantic(pseudo_instances, hs, ws, folder_name, file_id, local_iter, flag='pseudo')
    if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
        t_instances_img = 255 * np.ones(target_img.shape, dtype=np.uint8)
    #     target_img_vis = target_img.cpu().permute(1,2,0).numpy()
    #     target_img_vis = cv2.cvtColor(target_img_vis,cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_target_ori.jpg', target_img_vis)

    pred_instance_select = Instances((hs, ws))
    THIS_FRAME_HAS_RARE_CLASSES = False
    for i, obj_mask in enumerate(pred_masks.cpu()):
    # for i in range(gt_masks.shape[0]):
        instance_size = (obj_mask).sum().item()
        if instance_size == 0:
            continue 
        obj_mask = obj_mask.bool()
        # x_shift, y_shift = get_object_shift_by_depth_map(obj_mask, depth_map_target, depth_map_source) # TODO SHIFT FOR THIS
        x_shift, y_shift = 0, 0
        ''' shift obj_mask'''
        # x_shift = random.randint(-250, 250)
        # if instance_size < 5000: 
        #     y_shift = random.randint(-100, -50)
        # else:
        #     y_shift = random.randint(0, 100)
        
        shift_obj_mask, shift_target_image = translated_obj_mask(obj_mask,target_img, dx=x_shift,dy=y_shift)    

        ins = Instances((hs, ws))
        ins.gt_classes = pred_classes[i].cpu().view(1)
        ins.gt_masks = shift_obj_mask.cpu().view(1, hs, ws)
        ''' gather all big source instances'''
        if len(pred_instance_select._fields) == 0:
            pred_instance_select = Instances.cat([ins, ins]) # first one is redandence
        else:
            pred_instance_select = Instances.cat([pred_instance_select, ins])
        ''' mix the image'''
        for c in range(3):
            source_img[c,:][shift_obj_mask] = shift_target_image[c,:][shift_obj_mask]
            if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
                t_instances_img[c,:][shift_obj_mask] = shift_target_image[c,:][shift_obj_mask]
        # # for rare class balance
        # if pred_classes[i].item() in RARE_CLASS_NAMES:
        #     target_rare_class_samples.append({'img':source_img, 'instance':ins})
        #     THIS_FRAME_HAS_RARE_CLASSES = True

    if len(pred_instance_select._fields) != 0: # if gt_instance_selecthas nothing, no labels mix
        # pred_instance_select_rename = Instances((hs, ws))
        # pred_instance_select_rename.gt_masks = pred_instance_select.pred_masks.bool().cpu()
        # pred_instance_select_rename.gt_classes = pred_instance_select.pred_classes.cpu()

        # # do class balance
        # if not THIS_FRAME_HAS_RARE_CLASSES and len(target_rare_class_samples):
        #     # print('rare class sample : ', len(target_rare_class_samples))
        #     pred_instance_select, target_rare_class_samples = rare_class_balance(target_rare_class_samples, source_img, pred_instance_select)

        remove_occlussion(gt_instance, pred_instance_select[1:]) # modify pseudo_instances, remove parts of coverd by gt_instance_select
        one_data['source']['instances'] = Instances.cat([gt_instance, pred_instance_select[1:]])
        one_data['source']['image'] = source_img
        if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
            color_instances = visulize_color_instances(one_data['source']['instances'])
            color_pseudo_instances = visulize_color_instances(pseudo_instances)
            
            target_img_vis = target_img.cpu().permute(1,2,0).numpy()
            source_img_vis = source_img.cpu().permute(1,2,0).numpy()
            target_img_vis = cv2.cvtColor(target_img_vis,cv2.COLOR_BGR2RGB)
            source_img_vis = cv2.cvtColor(source_img_vis,cv2.COLOR_BGR2RGB)
            source_img_vis_cp = copy.deepcopy(source_img_vis)
            if VISUALIZE_POLYGON:   
                color_map = get_cityscapes_labels()
                for i in range(one_data['source']['instances'].__len__()):
                    mask = one_data['source']['instances'].gt_masks[i].to(torch.uint8).numpy()
                    class_id = one_data['source']['instances'].gt_classes[i].item()
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        cv2.drawContours(source_img_vis_cp, [contour], -1, (color_map[class_id][2],color_map[class_id][1], color_map[class_id][0]), 2)  # 轮廓颜色根据CityScapes 的颜色
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_t2s_mix.jpg', source_img_vis_cp)
            t_instances_img = t_instances_img.transpose((1,2,0))
            t_instances_img = cv2.cvtColor(t_instances_img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_t2s_instance.jpg' , t_instances_img)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_t2s_color_instance.jpg', color_instances)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_pseudo_color_instance.jpg', color_pseudo_instances)
            del source_img_vis_cp
        del pseudo_instances

            # cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_gt_img.jpg', source_img_vis)
            # instance_poly2color_semantic(pseudo_instances, hs, ws, folder_name, file_id, local_iter, flag='mix')
    # return one_data, target_rare_class_samples
    return one_data


def  dynamic_threshold_by_size(size_obj, image_size):
    ''' this paramters make threshold in 0.75-0.95'''
    base_size = image_size/10
    normalized_area = 1 if (size_obj / base_size) > 1 else (size_obj / base_size)
    # 设定参数
    d = 0.75
    b = 1
    a = 0.2 / np.log(b + 1)
    return a * np.log(b * normalized_area + 1) + d



def refine_class_combine_clip_m2f(clip_result, m2f_result, update_flag=False): # instance_mask_size, image_shape,
    ''' combine CLIP with mask2former result, 
    if m2f output score<thre and clip is confident, use clip, 
    then if clip output is beyond CITYSCAPES_THING_CLASSES, make m2f score low. 
    while if in CITYSCAPES_THING_CLASSES, use clip result and update class and score of m2f '''
    clip_class, clip_score, clip_probs = clip_result[0], clip_result[1], clip_result[2], 
    m2f_class, m2f_score, m2f_probs = m2f_result[0], m2f_result[1], m2f_result[2]

    if m2f_class == clip_class:
        return m2f_class, 1.0, update_flag # pretty sure about class
    else:#if not the same class,update
        # c, w, h = image_shape
        # image_size = w * h
        # # m2f_score_threshold = dynamic_threshold_by_size(instance_mask_size, image_size)
        # m2f_score_threshold = 0.9
        # if m2f_score < m2f_score_threshold and clip_score > 0.5:
        #     update_class = True
        #     return clip_class, m2f_score*1.2 if m2f_score*1.2<1.0 else 1.0, update_class
        # else:
        #     return m2f_class, m2f_score, update_class
        update_flag = True
        combined_class, combined_score = combine_clip_m2f_result(clip_probs, m2f_probs)
        return combined_class, combined_score, update_flag

def bar_chart_probs(classes, probs, save_name):
    ''' generate bar chart for the probs'''
    plt.figure(figsize=(10, 6))
    plt.bar(classes, probs, color='skyblue')
    plt.title('Probabilities for Each Category')
    plt.xlabel('Category')
    plt.ylabel('Probability')
    plt.xticks(rotation=90)
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(save_name.replace('.png', '_bar_chart.png'))
    plt.close()  # 关闭图表以释放内存
    
def entropy(p):
    if p.device.type == 'cpu':
        p = p.cuda() 
    return -torch.sum(p * torch.log(p + 1e-15))


def combine_clip_m2f_result(clip_probs, mask2former_probs):
    # multiply the two probabilities
    # TODO: m2f overwhelmingly confident, clip is not confident, use model
    if 0:
        combined_probs = clip_probs * mask2former_probs
    if 0:
        # normalize the probabilities
        logits_a = np.log(clip_probs + 1e-15)  # 加上微小值防止 log(0)
        logits_b = np.log(mask2former_probs + 1e-15)
        combined_logits = logits_a + logits_b
        combined_probs = np.exp(combined_logits)
    if 0:
        # 计算模型 A 和模型 B 的熵值
        entropy_a = entropy(clip_probs).cpu()
        entropy_b = entropy(mask2former_probs).cpu()
        
        # 计算平滑因子 alpha
        alpha = entropy_a / (entropy_b + 1e-15)
        
        # 对模型 B 的概率进行调整
        adjusted_probs_b = torch.pow(mask2former_probs, alpha)
        adjusted_probs_b /= adjusted_probs_b.sum()
        
        # 重新进行概率乘积法
        combined_probs = clip_probs * adjusted_probs_b
    if 0:# too trust clip
        combined_probs = 0.7*clip_probs + 0.3*mask2former_probs
    if 1:
        # 获取排序的索引（从高到低）
        ranks_a = torch.argsort(-clip_probs)
        ranks_b = torch.argsort(-mask2former_probs)
        
        # 对每个类别计算平均排名
        avg_ranks = (ranks_a + ranks_b) / 2.0
        
        # 根据平均排名得到最终预测
        combined_probs = torch.zeros_like(clip_probs)
        combined_probs[0][torch.argmin(avg_ranks)] = 1.0  # 将概率置于 1.0
        return torch.argmin(avg_ranks), 1.0
        
        
    combined_probs /= combined_probs.sum()
    final_prediction_class = np.argmax(combined_probs)
    print('clip_probs : ', clip_probs, ',mask2former_probs : ', mask2former_probs)
    print('final_prediction_probs : ', combined_probs)
    return final_prediction_class, combined_probs.max()
    
def correct_label_by_GT(instances, image_path):
    ''' correct instance result according to semantic GT '''
    # image_path='/datafast/120-1/Datasets/segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'
    # semantic='/datafast/120-1/Datasets/segmentation/Cityscapes/gtFine_trainvaltest/gtFine/train/aachen/aachen_000012_000019_gtFine_labelTrainIds.png'
    semantic_label_path = image_path.replace('/leftImg8bit_trainvaltest/leftImg8bit/', '/gtFine_trainvaltest/gtFine/').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
    semantic_image = cv2.imread(semantic_label_path, cv2.IMREAD_GRAYSCALE)
    pred_masks = instances.pred_masks  # Tensor of shape [N, H, W]
    pred_classes = instances.pred_classes  # Tensor of shape [N]
    remap_dict = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7 }
    # print('init time : ', time.time() - s)
    # 遍历每个实例
    collect_class_correct_pair = []
    for idx in range(len(instances)):
        pred_mask = pred_masks[idx].bool().numpy() # Tensor of shape [H, W]
        # 将mask转换为numpy数组
        # pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)  # 值为0或1
        masked_values = semantic_image[pred_mask]
        top_three_values = Counter(masked_values).most_common(3)
        
        for k, v in top_three_values:
            if k in remap_dict:
                if pred_classes[idx] != remap_dict[k]:
                    print('change class from ', pred_classes[idx], ' to ', remap_dict[k])
                    collect_class_correct_pair.append((pred_classes[idx], remap_dict[k]))
                    pred_classes[idx] = remap_dict[k]
                break
    return instances, collect_class_correct_pair

def remove_empty_instance_by_GT(instances, image_path):
    ''' correct instance result according to semantic GT '''
    # image_path='/datafast/120-1/Datasets/segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'
    # semantic='/datafast/120-1/Datasets/segmentation/Cityscapes/gtFine_trainvaltest/gtFine/train/aachen/aachen_000012_000019_gtFine_labelTrainIds.png'
    semantic_label_path = image_path.replace('/leftImg8bit_trainvaltest/leftImg8bit/', '/gtFine_trainvaltest/gtFine/').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
    semantic_image = cv2.imread(semantic_label_path, cv2.IMREAD_GRAYSCALE)
    if semantic_image is None:
        raise FileNotFoundError(f"Semantic label file not found: {semantic_label_path}")
    pred_masks = instances.pred_masks  # Tensor of shape [N, H, W]
    num_instances = len(instances)
    keep = torch.ones(num_instances, dtype=torch.bool)
    # 遍历每个实例
    count_zero = 0
    for idx in range(len(instances)):
        pred_mask = pred_masks[idx].bool().numpy() # Tensor of shape [H, W]
        keep_flag = True
        # 将mask转换为numpy数组
        # pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)  # 值为0或1
        masked_values = semantic_image[pred_mask]
        
        top_two_values = Counter(masked_values).most_common(2)
        if len(top_two_values) == 0:
            keep_flag = False
            count_zero += 1
        keep[idx] = keep_flag
    instances = instances[keep]
    print('remove empty instances : ', num_instances - len(instances), ' , zero label instances : ', count_zero)
    return instances


def keep_stuff_label_instance_by_GT(instances, image_path):
    ''' correct instance result according to semantic GT '''
    # image_path='/datafast/120-1/Datasets/segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'
    # semantic='/datafast/120-1/Datasets/segmentation/Cityscapes/gtFine_trainvaltest/gtFine/train/aachen/aachen_000012_000019_gtFine_labelTrainIds.png'
    semantic_label_path = image_path.replace('/leftImg8bit_trainvaltest/leftImg8bit/', '/gtFine_trainvaltest/gtFine/').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
    semantic_image = cv2.imread(semantic_label_path, cv2.IMREAD_GRAYSCALE)
    if semantic_image is None:
        raise FileNotFoundError(f"Semantic label file not found: {semantic_label_path}")
    pred_masks = instances.pred_masks  # Tensor of shape [N, H, W]
    pred_classes = instances.pred_classes  # Tensor of shape [N]
    remap_dict = {11, 12, 13, 14, 15, 16, 17, 18 }
    num_instances = len(instances)
    keep = torch.ones(num_instances, dtype=torch.bool)
    # 遍历每个实例
    count_zero = 0
    for idx in range(len(instances)):
        pred_mask = pred_masks[idx].bool().numpy() # Tensor of shape [H, W]
        keep_flag = True
        # 将mask转换为numpy数组
        # pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)  # 值为0或1
        masked_values = semantic_image[pred_mask]
        top_two_values = Counter(masked_values).most_common(2)
        for k, v in top_two_values:
            if k in remap_dict:
                keep_flag = False
                break
        keep[idx] = keep_flag
    instances = instances[keep]
    print('keep stuff  label instances : ', num_instances - len(instances))
    return instances



def remove_wrong_label_instance_by_GT(instances, image_path):
    ''' correct instance result according to semantic GT '''
    # image_path='/datafast/120-1/Datasets/segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'
    # semantic='/datafast/120-1/Datasets/segmentation/Cityscapes/gtFine_trainvaltest/gtFine/train/aachen/aachen_000012_000019_gtFine_labelTrainIds.png'
    semantic_label_path = image_path.replace('/leftImg8bit_trainvaltest/leftImg8bit/', '/gtFine_trainvaltest/gtFine/').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
    semantic_image = cv2.imread(semantic_label_path, cv2.IMREAD_GRAYSCALE)
    if semantic_image is None:
        raise FileNotFoundError(f"Semantic label file not found: {semantic_label_path}")
    pred_masks = instances.pred_masks  # Tensor of shape [N, H, W]
    pred_classes = instances.pred_classes  # Tensor of shape [N]
    # remap_dict = {11, 12, 13, 14, 15, 16, 17, 18 }
    num_instances = len(instances)
    keep = torch.ones(num_instances, dtype=torch.bool)
    # 遍历每个实例
    count_zero = 0
    for idx in range(len(instances)):
        pred_mask = pred_masks[idx].bool().numpy() # Tensor of shape [H, W]
        masked_values = semantic_image[pred_mask]
        top_two_values = Counter(masked_values).most_common(1)
        if len(top_two_values) == 0:
            count_zero += 1
            keep[idx] = False
            continue
        # print('top_two_values : ', top_two_values)
        for k, v in top_two_values:
            if k<11 or k>18:
                keep[idx] = False
                print('remove ', k,v)
                break
    instances = instances[keep]
    print('remove wrong label instances : ', num_instances - len(instances), ' , zero label instances : ', count_zero)
    return instances

def remap_clip_class_2cityscapes(clip_class):
    ''' remap clip class to cityscapes class'''    
    if clip_class in ["Pedestrian", "walking people", "people"]:
        return "person"
    elif clip_class in ["rider", "riding people", "biker", "motorcyclist", "cyclist"]:
        return "rider"
    elif clip_class in ["car", "sedan", "a van truck", "wagon", "hatchback", "coupe", "convertible", "SUV", "crossover", "minivan", "MPV"]:
        return "car"
    elif clip_class in ["truck", "a box truck", "Tractor Truck", "Trailer Truck", "Pickup Truck", "Semi-trailer Truck", 
                        "Dump Truck", "Garbage Truck", "Fire Truck", "Tanker Truck", "Concrete Mixer Truck", "Refrigerator Truck", 
                        "Logging Truck", "Car Carrier Truck", "Flatbed Truck"]:
        return "truck"
    elif clip_class in ["bus", "public transport bus", "school bus", "minibus", "Ambulance", "trolley bus", "double-decker bus", "articulated bus", 
                        "shuttle bus", "tour bus", "party bus", "sightseeing bus", "airport bus", "intercity bus"]:
        return "bus"
    elif clip_class in ["train", "Tram","Metro",]:
        return "train"                      
    elif clip_class in ["Standard Motorcycle", "Scooter", "Moped", "Trike", "Chopper", "Bobber", "Cafe Racer", "Streetfighter",  "Motocross Bike", 
                        "Supermoto Bike", "a part of Motorcycle"]:
        return "motorcycle"
    elif clip_class in ["Road Bike", "Mountain Bike", "a part of bike wheel"]:
        return "bicycle"
        
        
def sum_probs_for_category(clip_probs, clip_text_prompt, group_names):
    # 计算每组类别的概率总和
    group_probs_sum = {}
    for group, names in group_names.items():
        indices = [clip_text_prompt.index(name) for name in names]
        indices_tensor = torch.tensor(indices, device=clip_probs.device)
        group_probs_sum[group] = torch.sum(clip_probs[0, indices_tensor]).item()
    max_key = max(group_probs_sum, key=group_probs_sum.get)
    return max_key
    
    
def correct_label_by_CLIP(instances, input_image, imagepath=None, map_save_folder=None, debug_vis=False): # TODO， init once; model_clip, preprocess_clip, text_inputs
    ''' 
    instances : Instances
    input_image : tensor, 3, h, w
    '''
    # s = time.time()
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

    # 定义文本标签
    CITYSCAPES_THING_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",]
    
    group_names = {
    "person": [ "walking people", "people"],
    "rider": ["rider", "riding people", "biker", "motorcyclist", "cyclist"],
    "car": ["car", "sedan", "wagon", "hatchback", "coupe", "convertible", "SUV", "crossover", "minivan", "MPV", "Ambulance", "a van"],
    "truck": ["truck",  "Tractor Truck", "Trailer Truck", "Pickup Truck", "Semi-trailer Truck", "a box truck",
                "Dump Truck", "Garbage Truck", "Fire Truck",  "Tanker Truck", "Concrete Mixer Truck", "Refrigerator Truck", 
                "Logging Truck", "Car Carrier Truck", "Flatbed Truck"],
    "bus": ["bus", "public transport bus", "school bus", "trolley bus", "double-decker bus", "articulated bus", 
            "shuttle bus", "tour bus", "sightseeing bus", "airport bus", "intercity bus"],
    "train": ["train", "Tram", "Metro"],
    "motorcycle": ["Standard Motorcycle", "Scooter", "Moped", "Trike", "Chopper", "Bobber", "Cafe Racer", 
                    "Streetfighter", "Motocross Bike", "Supermoto Bike","motorcycle"],
    "bicycle": ["Road Bike", "Mountain Bike", "a part of bike"],
    "stuff": ["object",],
    "envionment": ["road", "building", "sky", "tree", "grass", "sidewalk"]
    }
    clip_text_prompt = []
    for group, names in group_names.items():
        clip_text_prompt.extend(names)
    # CITYSCAPES_STUFF_CLASSES = [
    #     "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    #     "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    #     "truck", "bus", "train", "motorcycle", "bicycle"]
    text_inputs = clip.tokenize(clip_text_prompt).to(device)
    
    class_scores = instances.class_scores  # Tensor of shape [N]
    mask_scores = instances.mask_scores  # Tensor of shape [N]
    # scores = instances.scores  # Tensor of shape [N]
    scores_8s = instances.scores_8  # Tensor of shape [N]
    pred_masks = instances.pred_masks  # Tensor of shape [N, H, W]
    pred_classes = instances.pred_classes  # Tensor of shape [N]
    # print('init time : ', time.time() - s)
    # 遍历每个实例
    keep = torch.ones(len(instances), dtype=torch.bool)
    for idx in range(len(instances)):
        # class_score = class_scores[idx]
        pred_mask = pred_masks[idx]  # Tensor of shape [H, W]
        # 将mask转换为numpy数组
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)  # 值为0或1
        # 定义形态学操作的核
        kernel = np.ones((3, 3), np.uint8)

        # 进行形态学腐蚀处理
        eroded_mask = cv2.erode(pred_mask_np, kernel, iterations=1)

        # 可选：去掉小点
        # 你可以使用形态学开运算（先腐蚀后膨胀）来去掉小点
        pred_mask_np = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, kernel)


        # 找到mask的非零区域，计算边界框
        coords = np.column_stack(np.where(pred_mask_np > 0))
        if coords.size == 0:
            continue  # 如果掩码为空，跳过
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # 在原始图像和掩码上裁剪边界框区域
        
        if isinstance(input_image, torch.Tensor):
            image_np = input_image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = input_image
        image_np = image_np.astype(np.uint8)

        cropped_image = image_np[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = pred_mask_np[y_min:y_max+1, x_min:x_max+1]
        ''' only update object of size > 10000'''
        if cropped_mask.sum() < 5000:
            continue
        if cropped_mask.sum() / (cropped_image.size/3) < 0.8:
            continue

        # 应用掩码到裁剪后的图像上
        cropped_mask_3d = cropped_mask[:, :, np.newaxis]
        masked_image_array = cropped_image * cropped_mask_3d

        # 将结果转换回PIL图像
        masked_image = Image.fromarray(masked_image_array.astype('uint8'))
        cropped_image = Image.fromarray(cropped_image.astype('uint8'))

        # 对图像进行CLIP预处理
        image_input = preprocess_clip(cropped_image).unsqueeze(0).to(device)
        # 计算图像和文本的特征
        with torch.no_grad():
            # image_features = model_clip.encode_image(image_input)
            # text_features = model_clip.encode_text(text_inputs)

            # 计算相似度
            logits_per_image, _ = model_clip(image_input, text_inputs)

            probs = logits_per_image.softmax(dim=-1).cpu()
            # clip_var = np.var(probs)

        # 获取预测的类别
        clip_predicted_class = clip_text_prompt[torch.argmax(probs).item()]
        # predicted_class = remap_clip_class_2cityscapes(clip_predicted_class)
        predicted_class = sum_probs_for_category(probs, clip_text_prompt, group_names)

        ''' generate bar chart for the probs'''
        if imagepath:
            image_name = imagepath.split('/')[-1]
        else:
            image_name = '.png'
        save_name = map_save_folder + '/' + image_name
        
        if debug_vis:
            bar_chart_probs(clip_text_prompt, probs[0], save_name.replace('.png', '_clip.png'))
            bar_chart_probs(CITYSCAPES_THING_CLASSES, scores_8s[idx], save_name.replace('.png', '_m2f.png'))
        
        ''' combine CLIP with mask2former result, 
        if m2f output score<thre and clip is confident, use clip, 
        then if clip output is beyond CITYSCAPES_THING_CLASSES, make m2f score low. 
        while if in CITYSCAPES_THING_CLASSES, use clip result and update class and score of m2f '''
        # update_class_label, update_class_score = combine_clip_m2f_result(probs, scores_8s[idx])
        # clip_result = [torch.argmax(probs), probs.max().item(), probs]
        # m2f_result = [pred_classes[idx], class_score.item(), scores_8s[idx]]
        
        m2f_result_class = CITYSCAPES_THING_CLASSES[pred_classes[idx]]
        if m2f_result_class == predicted_class:
            class_scores[idx] = 1.0
        elif  m2f_result_class!= predicted_class and predicted_class not in ['stuff', 'envionment']:
            # print('CLIP, ', predicted_class, 'm2f, ', m2f_result_class, imagepath)
            update_class_label = torch.tensor(CITYSCAPES_THING_CLASSES.index(predicted_class))
            pred_classes[idx] = update_class_label
        elif predicted_class in ['stuff', 'envionment']:
            keep[idx] = False
        
        if debug_vis:
            image_name_pre = image_name.split('.')[0]
            save_name = map_save_folder + '/' + image_name_pre + '_m2f-' + m2f_result_class + '-map-' + predicted_class + '_clip-' + clip_predicted_class + '.jpg'
            cropped_image.save(save_name)
            masked_image.save(save_name.replace('.jpg', '_mask.jpg'))
        #TODO if predicted_class  in ['stuff', 'envionment'], remove
            
            
        # if CITYSCAPES_THING_CLASSES[pred_classes[idx]] in ['person', 'rider'] or CITYSCAPES_THING_CLASSES[clip_result[0]] in ['person', 'rider']:
        #     continue
        
        # # ''' 1. first update class score, 2. to think about updating the mask score'''
        # update_class_label, update_class_score, update_class = refine_class_combine_clip_m2f(clip_result, m2f_result)
        # if CITYSCAPES_THING_CLASSES[pred_classes[idx]] !=  CITYSCAPES_THING_CLASSES[update_class_label]:
        #     print(f"CLIP,  {probs.max():.2f}, {predicted_class},  m2f, {class_score.item():.2f}, {CITYSCAPES_THING_CLASSES[pred_classes[idx]]}")
        #     print(f"update class from {CITYSCAPES_THING_CLASSES[pred_classes[idx]]} to {CITYSCAPES_THING_CLASSES[update_class_label]}")
        #     save_name = 'output/clip_class_refine/update_size_' + str(int(cropped_mask.sum())) + '-' + CITYSCAPES_THING_CLASSES[pred_classes[idx]] + '-to-' + CITYSCAPES_THING_CLASSES[update_class_label] + '.png'
        #     cropped_image.save(save_name)
        #     # masked_image.save(save_name.replace('.png', '_mask.png'))
            
        # # update class name and class score
        # # update_class_label_index = CITYSCAPES_THING_CLASSES.index(update_class_label)


    # after update each instance class and score, update the total scores
    instances.scores = class_scores*mask_scores
    
    instances = instances[keep]
    #TODO remove small instance with keep mask
    return instances
