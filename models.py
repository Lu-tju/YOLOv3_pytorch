import os

from utils.parse_config import *
from utils.utils import *

ONNX_EXPORT = False


def create_modules(module_defs):
    """
   构建网络模型
    """
    hyperparams = module_defs.pop(0)    # 超参数
    output_filters = [int(hyperparams['channels'])]     # 初始输入通道数(3)
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        # 构建卷积层模块：包含Conv+BN+leaky，这里先提取卷积参数，直接用nn模块构建层结构
        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        # yolo tiny是包含池化层的，也是先提取参数，再直接用nn.MaxPool2d构建
        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        # 创建上采样层，这里由于nn.upsample报警告，所以自己创建了Upsample函数
        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: 不推荐nn.Upsample这种方式
            upsample = Upsample(scale_factor=int(module_def['stride']))     # scale_factor：尺度缩放因子
            modules.add_module('upsample_%d' % i, upsample)

        # 创建路由层：空层，具体功能在前向传播中实现
        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        # 创建shortcut层：空层，具体功能在前向传播中实现
        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        # 创建yolo层，yolo层就是结构变换，把输出结构整理一下
        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]   # anchor_idxs = [6, 7, 8] 第几个anchor
            # 提取 anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]     # anchors = [(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)]
            nC = int(module_def['classes'])  # number of classes
            img_size = int(hyperparams['height'])
            # Define detection layer,初始化一下，重点实现在forward中（猜的）
            yolo_layer = YOLOLayer(anchors, nC, img_size, yolo_layer_count, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # 链接module列表，并存放各层卷积核个数
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list     # 返回超参数和网络列表


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # 自定义了Upsample层（nn.Upsample给出不推荐的警告消息）

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)     # interpolate：pytorch的上采样函数，默认插值：nearest


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, img_size, yolo_layer, cfg):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.img_size = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        create_grids(self, 32, 1, device=device)    # 没搞懂干了些啥

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_layer]  # stride of this layer
            if cfg.endswith('yolov3-tiny.cfg'):
                stride *= 2

            nG = int(img_size / stride)  # number grid points 416/32=13
            create_grids(self, img_size, nG)

    def forward(self, p, img_size, var=None):   # p：前层输出[batch, 225, 13, 13]
        if ONNX_EXPORT:
            bs, nG = 1, self.nG  # batch size, nG = grid size
        else:
            bs, nG = p.shape[0], p.shape[-1]    # batch size, grid size   bs=1,nG=13
            if self.img_size != img_size:       # self.img_size = 32 img_size = 416
                create_grids(self, img_size, nG, p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction——>1*3*13*13*85

        if self.training:
            return p

        elif ONNX_EXPORT:
            grid_xy = self.grid_xy.repeat((1, self.nA, 1, 1, 1)).view((1, -1, 2))
            anchor_wh = self.anchor_wh.repeat((1, 1, nG, nG, 1)).view((1, -1, 2)) / nG

            # p = p.view(-1, 5 + self.nC)
            # xy = xy + self.grid_xy[0]  # x, y
            # wh = torch.exp(wh) * self.anchor_wh[0]  # width, height
            # p_conf = torch.sigmoid(p[:, 4:5])  # Conf
            # p_cls = F.softmax(p[:, 5:], 1) * p_conf  # SSD-like conf
            # return torch.cat((xy / nG, wh, p_conf, p_cls), 1).t()

            p = p.view(1, -1, 5 + self.nC)
            xy = torch.sigmoid(p[..., 0:2]) + grid_xy  # x, y
            wh = torch.exp(p[..., 2:4]) * anchor_wh  # width, height
            p_conf = torch.sigmoid(p[..., 4:5])  # Conf
            p_cls = p[..., 5:]
            # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
            # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
            p_cls = torch.exp(p_cls).permute((2, 1, 0))
            p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
            p_cls = p_cls.permute(2, 1, 0)
            return torch.cat((xy / nG, wh, p_conf, p_cls), 2).squeeze().t()

        else:  # inference
            p[..., 0:2] = torch.sigmoid(p[..., 0:2]) + self.grid_xy  # 公式bx=sigmoid(tx)+cx, by=sigmoid(ty)+cy
            p[..., 2:4] = torch.exp(p[..., 2:4]) * self.anchor_wh  # 公式bw=pw×e^tw, bh=ph×e^th
            # p[..., 2:4] = ((torch.sigmoid(p[..., 2:4]) * 2) ** 2) * self.anchor_wh  # wh power method
            p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf=sigmoid(conf) bbox的置信度
            p[..., :4] *= self.stride   # 映射回416*416

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85] (507=3*13*13，即所有的bbox)
            return p.view(bs, -1, 5 + self.nC)


class Darknet(nn.Module):
    """YOLOv3目标检测模型，实现前向传播"""

    def __init__(self, cfg_path, img_size=416):     # 模型和参数初始化
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)   # 模型构建
        self.img_size = img_size
        self.loss_names = ['loss', 'xy', 'wh', 'conf', 'cls', 'nT']
        self.losses = []

    def forward(self, x, var=None):     # 前向传播
        img_size = x.shape[-1]     # 416
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']     # module_def层信息：Conv、filter、pad... module:层结构
            # 'convolutional', 'upsample', 'maxpool'层直接输入层结构
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            # 'route' 'shortcut' 'yolo'层单独写的运算
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, img_size)  # 进入yolo层的前向传播，x:[1,507,85]每行一个bbox信息
                output.append(x)
            layer_outputs.append(x)     # 储存中间层输出

        if ONNX_EXPORT:
            output = torch.cat(output, 1)  # merge the 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            return output[5:].t(), output[:4].t()  # ONNX scores, boxes
        else:
            return output if self.training else torch.cat(output, 1)    # 整合3个尺度， 85 x (507, 2028, 8112) ——> 85 x 10647


def get_yolo_layers(model):
    a = [module_def['type'] == 'yolo' for module_def in model.module_defs]
    return [i for i, x in enumerate(a) if x]  # [82, 94, 106] for yolov3


def create_grids(self, img_size, nG, device='cpu'):
    ''' 生成gird坐标，用于还原预测的绝对位置 '''
    self.img_size = img_size
    self.stride = img_size / nG     # 缩放倍数，用于anchor相对grid归一化

    # build xy offsets，这个5维是为了对应输出[1,3,13,13,85]:[batch,num_anchor,grid_x,grid_y,80类+位置+conf]
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4).to(device)     # Size：[1,1,13,13,2]:对应每个grid左上角位置，由于预测框是基于gird的，需要最后再加上grid的位置，才是绝对位置

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride     # Size:[3,2] 三行各对应3个anchor相对gird的大小（归一化）
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2).to(device)   # Size:[1,3,1,1,2],
    self.nG = torch.FloatTensor([nG]).to(device)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75     # yolo v3结构中，Darknet53为0-75层（除掉分类层）
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15     # yolo tiny结构中，backbone为0-15层

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        # 就只有卷积模块有参数，分别是Conv和BN层
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
