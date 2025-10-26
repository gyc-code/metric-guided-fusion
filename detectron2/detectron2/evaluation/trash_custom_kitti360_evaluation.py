# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import logging
import numpy as np
import cv2
import os
import sys
import tempfile
from collections import OrderedDict
import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils import comm
from .evaluator import DatasetEvaluator
from detectron2.utils.visualizer import Visualizer, ColorMode
from .custom_cityscapes_evaluation import CityscapesCustomEvaluator
import shutil

# Print an error message and quit
def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)
    


class Kitti360CustomInstanceEvaluator(CityscapesCustomEvaluator):
    """
    Evaluate instance segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """
    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._working_dir, "predictions", "result", basename + "_pred.txt")
            if not os.path.exists(os.path.join(self._working_dir, "predictions", "result")):
                os.makedirs(os.path.join(self._working_dir, "predictions", "result"))
            if not os.path.exists(os.path.join(self._working_dir, "predictions", "instance_visualization")):
                os.makedirs(os.path.join(self._working_dir, "predictions", "instance_visualization"))
            if self.save_instances:
                self.cs_metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
                im = read_image(file_name, format="BGR")
                v = Visualizer(im[:, :, ::-1], self.cs_metadata, scale=1.2, instance_mode=ColorMode.IMAGE)
                instance_result = v.draw_instance_predictions(output["instances"].to(self._cpu_device))
                instance_result.save(self._working_dir + os.sep + "predictions" + os.sep + "instance_visualization" + os.sep +  basename + ".png")
            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.thing_classes[pred_class]
                        class_id = name2label[classes].id
                        score = output.scores[i]
                        mask = output.pred_masks[i].numpy().astype("uint8")
                        png_filename = os.path.join(
                            self._working_dir, "predictions", "result", basename + "_{}_{}.png".format(i, classes)
                        )
                        Image.fromarray(mask * 255).save(png_filename)

                        fout.write(
                            "{} {} {}\n".format( os.path.basename(png_filename), class_id, score)
                        )
            else:
                # Cityscapes requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import kitti360scripts.evaluation.semantic_2d.evalInstanceLevelSemanticLabeling as kitti360_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        kitti360_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        kitti360_eval.args.predictionWalk = None
        kitti360_eval.args.JSONOutput = False
        kitti360_eval.args.colorized = False
        kitti360_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances_kitti360.json")
        

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        
        predictionImgList = []
        groundTruthImgList = []
        # we support the no-argument way only as the groundTruthImgList should contain paths to both semantic and confidence maps
        
        # args.groundTruthListFile = os.path.join(args.kitti360Path, 'data_2d_semantics', 'train', '2013_05_28_drive_val_frames.txt')
        groundTruthListFile = '/home/yguo/Documents/other/UDA4Inst/datasets/kitti360/2013_05_28_drive_val_frames_all.txt'
        # use the ground truth search string specified above
        groundTruthImgList = kitti360_eval.getGroundTruth(groundTruthListFile)
        if not groundTruthImgList:
            printError("Cannot find any ground truth images to use for evaluation.")
        # get the corresponding prediction for each ground truth imag
        for gt,_ in groundTruthImgList:
            predictionImgList.append( kitti360_eval.getPrediction(kitti360_eval.args, gt) )

        # print some info for user
        print("Note that this tool uses the file '{}' to cache the ground truth instances.".format(kitti360_eval.args.gtInstancesFile))
        print("If anything goes wrong, or if you change the ground truth, please delete the file.")

            
        try:
            results = kitti360_eval.evaluateImgLists(
                predictionImgList, groundTruthImgList, kitti360_eval.args
            )["averages"]

            ret = OrderedDict()
            ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
            self._logger.info(results)
            # self._working_dir.cleanup()
            shutil.rmtree(os.path.abspath(self._working_dir + os.sep + "predictions" + os.sep + "result"))
            return ret
        except:
            print('------------ error happen in eval')
            self._working_dir.cleanup() 
            return None


