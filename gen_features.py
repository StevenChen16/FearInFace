import argparse
import os
import time
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel

from models.experimental import attempt_load
from models.resnet import resnet_face18
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging

from utils.torch_utils import select_device, time_synchronized


def gen(opt):
    source, yolo_weight, imgsz, arcface_weight, out_dir = opt.source, opt.yolo_weight, opt.img_size, opt.arcface_weight, opt.out_dir
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weight, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    modelc = resnet_face18(False)
    modelc = DataParallel(modelc)
    modelc.load_state_dict(torch.load(arcface_weight, map_location=device), strict=False)
    modelc.to(device)
    modelc.eval()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # crop
                    face_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    # recognition
                    face_img = cv2.resize(face_img, (128, 128))

                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                    face_img = np.dstack((face_img, np.fliplr(face_img)))

                    face_img = face_img.transpose((2, 0, 1))
                    face_img = face_img[:, np.newaxis, :, :]

                    face_img = face_img.astype(np.float32, copy=False)
                    face_img -= 127.5
                    face_img /= 127.5

                    face_data = torch.from_numpy(face_img)
                    face_data = face_data.to(device)

                    output = modelc(face_data)  # 获取特征
                    output = output.data.cpu().numpy()

                    fe_1 = output[::2]
                    fe_2 = output[1::2]

                    feature = np.hstack((fe_1, fe_2))
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    flie_name = os.path.join(out_dir, os.path.split(p)[1].split(".")[0])
                    if os.path.exists(flie_name + ".npy"):
                        append_data(flie_name=flie_name, feature=feature)
                    else:
                        np.save(flie_name, [feature])

    print(f'Done. ({time.time() - t0:.3f}s)')
    print("generate feature sucessful!")


def append_data(flie_name, feature):
    data = np.load(flie_name + ".npy")
    for i in data:
        if (i == feature).all:
            return
    data.append(feature)
    np.save(flie_name, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='person', help='out_dir')
    parser.add_argument('--yolo_weight', type=str, default='weights/yolo_face.pt', help='model.pt path(s)')
    parser.add_argument('--arcface_weight', type=str, default=r'weights/arcface.pth',
                        help='arcface.pth path')
    parser.add_argument('--source', type=str, default='images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    gen(opt=opt)
