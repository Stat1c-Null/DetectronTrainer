#Verify that detectron2 got installed
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
print("Detectron installed")

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda) 

# Check if Detectron2 is properly installed

# Register your dataset
from detectron2.data.datasets import register_coco_instances


register_coco_instances("car_dataset_train", {}, "dataset/annotations/instances_train2017.json", "dataset/train2017")
register_coco_instances("car_dataset_val", {}, "dataset/annotations/instances_val2017.json", "dataset/val")


print("registered dataset")

#Configure model
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("configs/config.yaml")  # Load a config file
cfg.DATASETS.TRAIN = ("car_dataset_train",)
cfg.DATASETS.TEST = ("car_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "model_final_721ade.pkl"  # Initialize from a pretrained model
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Number of classes in your dataset

print("Configured the model")

#Train Model
print("Training the model")
if __name__ == "__main__":
  from detectron2.engine import DefaultTrainer

  trainer = DefaultTrainer(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()

print("Finished training the model")

#Evaluate the model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("car_dataset_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "car_dataset_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

print("Evaluated the model")

#Run inference on a new image
from detectron2.engine import DefaultPredictor
import cv2

# Load the model
predictor = DefaultPredictor(cfg)

# Load an image
im = cv2.imread("test/street.jpg")

# Run inference
outputs = predictor(im)

# Visualize the results
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
cv2.waitKey(0)