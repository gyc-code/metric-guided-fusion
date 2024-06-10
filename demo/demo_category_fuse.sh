
CUDA_VISIBLE_DEVICES=1 python demo_catogory_fuse.py \
--config-file configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml \
--input  /home/yguo/Documents/other/datasets/cityscapes/leftImg8bit/val \
--output /home/yguo/Documents/other/Mask2Former/visual_instance/uda_synscapes_clean2_1024_bs3from_coco_huamn_cycle+vehicle_t2s_s2t \
--opts_human_cycle MODEL.WEIGHTS './output/uda_synscapes_clean2_1024_bs3from_coco_huamn_cycle_t2s_s2t_motor_augum/model_best.pth'\
--opts_vehicle MODEL.WEIGHTS './output/uda_synscapes_clean2_1024_bs3from_coco_vehicle_t2s_s2t_train_augum/model_best.pth'
