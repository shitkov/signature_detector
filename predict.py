import cv2
import numpy as np
import math
import torch
import torchvision

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox

@torch.no_grad()
def predict(
        weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=[1024, 1024],  # inference size (pixels, WxH)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        ):

    # Initialize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load FP32 model
    model = attempt_load(weights, map_location=device, fuse=False)
    # model stride
    stride = int(model.stride.max())
    # get class names
    names = model.names

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    im0s = cv2.imread(source)
    # Convert
    img = letterbox(im0s, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    pred = model(img, augment=augment, visualize=visualize)[0]
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    pred_list = []
    # Process predictions
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s.copy()

        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0  # for save_crop
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                pred_list.append([int(i) for i in np.array(xyxy)])
    return pred_list
