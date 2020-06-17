import argparse
import shutil
import time
from pathlib import Path
from sys import platform
import pickle as pkl
from models import *
from utils.datasets import *
from utils.utils import *

def write(x, results):
    img = results
    color = tuple([0, 255,  255])
    # 画bbox框
    tl = round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)


def detect(
        cfg,
        weights,
        video,
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
):
    device = torch_utils.select_device()  # CUDA 0
    # Initialize model
    model = Darknet(cfg, img_size)      # 模型和参数初始化
    inp_dim =416

    # Load weights(对应.pt或.weight)
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)    # 对应模型,导入网络权重

    model.to(device).eval()     # 测试模式(dropout和batch normalization的操作在训练和测试时不一样)

    # 保存配置
    videofile = video  # or path to the video file.
    cap = cv2.VideoCapture(videofile)  # 用 OpenCV 打开视频
    frames = 0
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 24
    start = time.time()
    savedPath = './output/savevideo.avi'  # 保存的地址和视频名
    ret, frame = cap.read()
    videoWriter = cv2.VideoWriter(savedPath, fourcc, fps, (frame.shape[1], frame.shape[0]))  # 最后为视频图片的形状
    Size1 = tuple([720, 1280, 3])

    while cap.isOpened():  # ret指示是否读入了一张图片，为true时读入了一帧图片
        ret, frame = cap.read()  # 读入成功，ret=1，frame为图片

        if ret:
            # 将图片按照比例缩放缩放，将空白部分用(128,128,128)填充，得到为416x416的图片。并且将HxWxC转换为CxHxW
            # Padded resize——>416*416*3
            img, _, _, _ = letterbox(frame, height=416)  # 通过补灰，等比例调整成416*416

            # 进行格式转换
            img = img[:, :, ::-1].transpose(2, 0, 1)  # 416*416*3——>3*146*416，BGR——>RGB
            img = np.ascontiguousarray(img, dtype=np.float32)  # 变为连续存储的数组，因为进行转置操作后，图片相邻元素 对应的存储位置有了跳跃
            img /= 255.0  # 归一化
            img = torch.from_numpy(img).unsqueeze(0).to(device)  # 添加batch维度：1*3*416*416，并由numpy数组转换为tensor

            im_dim = frame.shape[1], frame.shape[0]  # 保存原始大小：(640, 480)
            # 先将im_dim变成长度为2的一维行tensor，再在1维度(列这个维度)上复制一次，变成1x4的二维行tensor[W,H,W,H]，展开成1x4主要是在后面计算x1,y1,x2,y2各自对应的缩放系数时好对应上。
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)  # 重复一次 [640,480,640,480]对应[x1,y1,x2,y2]
            im_dim = im_dim.to(device)

            output = model(img)  # 1*10647*85
            output = output[output[:, :, 4] > conf_thres]  # 去掉 conf < threshold 的bbox ——>[14,85]
            # output的正常输出类型为float32,如果没有检测到目标时output元素为0，此时为int型，将会用continue进行下一次检测
            # 如果没有对象
            if len(output) == 0:
                # 每次迭代，我们都会跟踪名为frames的变量中帧的数量。然后我们用这个数字除以自第一帧以来过去的时间，得到视频的（平均） 帧率。
                frames += 1
                print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
                # 我们不再使用cv2.imwrite将检测结果图像写入磁盘，而是使用cv2.imshow展示画有边界框的帧。
                cv2.imshow("frame", frame)
                videoWriter.write(frame)  # 每次循环，写入该帧
                key = cv2.waitKey(1)
                # 如果用户按Q按钮，就会让代码中断循环，并且视频终止。
                if key & 0xFF == ord('q'):
                    break
                continue  # 不再执行以下，重新读图

            ''' 坐标转换为实际原图坐标'''

            # NMS
            output = non_max_suppression(output.unsqueeze(0), conf_thres, nms_thres)[0]  # 目标数n*8

            # 将bbox位置从416*416映射回原始大小和比例下的位置
            scale_coords(img_size, output[:, :4], Size1).round()


            # 将每个方框的属性写在图片上
            list(map(lambda x: write(x, frame), output))

            cv2.imshow("frame", frame)

            videoWriter.write(frame)  # 每次循环，写入该帧
            key = cv2.waitKey(1)
            # 如果有按键输入则返回按键值编码，输入q返回113
            if key & 0xFF == ord('q'):
                break
            # 统计已经处理过的帧数
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            videoWriter.release()  # 未读入图片，即视频处理完了，结束循环的时候释放
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='cfg文件路径')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='权重文件路径')    # TODO 改
    parser.add_argument('--video', type=str, default='data/samples/video.mp4', help='视频文件')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.video,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )