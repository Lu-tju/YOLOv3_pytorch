import argparse
import shutil
import time
from pathlib import Path
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()    # CUDA 0
    if os.path.exists(output):
        shutil.rmtree(output)  # 删除output文件夹
    os.makedirs(output)  # 新建output文件夹

    # Initialize model
    model = Darknet(cfg, img_size)      # 模型和参数初始化

    # Load weights(对应.pt或.weight)
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)    # 对应模型,导入网络权重

    model.to(device).eval()     # 测试模式(dropout和batch normalization的操作在训练和测试时不一样)

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)  # 初始化dataloader，包括路径、文件数、（网路设定的）图像大小

    # Get classes and colors，得到颜色对，每一种类别，对应一个颜色
    classes = load_classes(parse_data_cfg('data/turtlebot.data')['names'])   # TODO 迁移学习记得改
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)     # 输出目录

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)   # 添加batch维度：1*3*416*416，并由numpy数组转换为tensor
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)   # pred：[batch, 10647, 85]
        pred = pred[pred[:, :, 4] > conf_thres]  # 去掉 conf < threshold 的bbox ——>[14,85]

        if len(pred) > 0:   # 若有目标
            # Run NMS on predictions,非极大值抑制NMS
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]   # [14,85]——>[3,7] 7为：4个位置,bbox_conf,类得分,哪一类

            # 将bbox位置从416*416映射回原始大小和比例下的位置
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen：1 bicycles, 1 trucks, 1 dogs
            unique_classes = detections[:, -1].cpu().unique()   # 挑出类别索引
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()  # 几个
                print('%g %ss' % (n, classes[int(c)]), end=', ')    # 几个什么

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])    # 画框，暂存到im0

        dt = time.time() - t
        print('Done. (%.3fs)' % dt)

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='cfg文件路径')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='权重文件路径')    # TODO 改
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
