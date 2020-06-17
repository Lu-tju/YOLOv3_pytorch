import argparse
import time

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,   # 重新开始
        epochs=10,
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,      # TODO 所有层参数都学习更新？
):
    weights = 'weights' + os.sep    # 路径：weights\
    latest = weights + 'latest.pt'  # 路径：weights\latest.pt
    best = weights + 'best.pt'      # # 路径：weights\best.pt
    device = torch_utils.select_device()

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
    else:
        torch.backends.cudnn.benchmark = True  # 不进行多尺度训练

    # 配置运行
    train_path = parse_data_cfg(data_cfg)['train']  # train_path= data\train.txt，存放训练集路径

    # Initialize model
    model = Darknet(cfg, img_size)  # 载入网络模型

    # Get dataloader
    dataloader = LoadImagesAndLabels(train_path, batch_size, img_size, augment=True)    # 创建dataloader，他包含训练集，label，batch等信息

    lr0 = 0.001  # initial learning rate
    cutoff = -1  # 不要darknet53最后一层? backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:      # 如果从头训练
        checkpoint = torch.load(latest, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])

        # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     p.requires_grad = True if (p.shape[0] == 255) else False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:   # 如果不是从头，即迁移学习，载入预训练的参数：
        # Initialize model with backbone (optional)
        if cfg.endswith('yolov3.cfg'):
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')
        elif cfg.endswith('yolov3-tiny.cfg'):   # 对应模型，加载模型权重
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')

        # Set optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=.9)    # 对model的parameters()采用SGD学习优化器

    if torch.cuda.device_count() > 1:   # 并行训练
        model = nn.DataParallel(model)
    model.to(device).train()

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    # Start training
    t0 = time.time()
    loss_list = []  # 画图用
    # model_info(model)     # 模型信息
    n_burnin = min(round(dataloader.nB / 5 + 1), 1000)  # 用于学习率的更新 number of burn-in batches
    for epoch in range(epochs):
        model.train()
        epoch += start_epoch    # 从最开始的epoch+这次的（如果之前训练过）

        print(('\n%8s%12s' + '%10s' * 7) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)
        if epoch > 250:     # 250个epoch学习率衰为0.0001
            lr = lr0 / 10
        else:
            lr = lr0
        for x in optimizer.param_groups:    # x是优化器的训练参数
            x['lr'] = lr

        # 冻结 backbone 参数，不进行更新（只学习yolo分类层）（在epoch 0冻结, 在epoch 1解冻）
        if freeze_backbone and epoch < 2:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # 冻结Darknet的层0-cutoff，只学习分类层参数 （yolo是75层，yolo tiny是15）
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            # imgs：4*3*416*416,（batch=4）    targets：目标数*6 6代表：[哪幅图的，类，4个位置]
            targets = targets.to(device)
            nT = targets.shape[0]   # 目标数
            if nT == 0:  # if no targets continue
                continue

            # lr从0开始逐渐增大到lr0，之后不再变化(开始太大会发散，不过迁移学习不用担心） （不过，lr应该开始大，越来越小才对呀）
            if (epoch == 0) and (i <= n_burnin):
                lr = lr0 * (i / n_burnin) ** 4
                # print(lr)
                for x in optimizer.param_groups:
                    x['lr'] = lr
            # Run model, 这里以yolo_tiny为例，它有两个尺度
            # pred：2个元组，对应2个尺度： [4,3,13,13,85] , [4,3,26,26,85]
            pred = model(imgs.to(device))

            # Build targets，转换训练需要格式，选取正负样本。
            # 包含5个元组，为txy, twh, tcls, tconf, indices
            target_list = build_targets(model, targets, pred)

            # 计算 loss
            loss, loss_dict = compute_loss(pred, target_list)   # loss.Size=1
            loss_list.append(loss.item())

            # Compute gradient
            loss.backward()

            # 梯度累计多少次进行一次参数更新 Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()   # 梯度清零

            # 计算每epoch平均误差并打印  Running epoch-means of tracked metrics
            ui += 1
            for key, val in loss_dict.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['total'],
                nT, time.time() - t0)
            t0 = time.time()
            print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataloader.img_size = random.choice(range(10, 20)) * 32
                # print('multi_scale img_size = %g' % dataloader.img_size)

        # Update best loss,这里应该用验证集来验证吧，防止过拟合，但实际上整个程序验证集并未用到
        if rloss['total'] < best_loss:
            best_loss = rloss['total']

        # 保存训练结果
        save = True
        if save:
            # Save latest checkpoint
            checkpoint = {'epoch': epoch,
                          'best_loss': best_loss,
                          'model': model.module.state_dict() if type(model) is nn.DataParallel else model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, latest)

            # Save best checkpoint
            if best_loss == rloss['total']:
                torch.save(checkpoint, best)

            # Save backup weights every 5 epochs (optional)
            if epoch > 0 and epoch % 5 == 0:
                torch.save(checkpoint, weights + 'backup%g.pt' % epoch)

        # Calculate mAP 其实mAP在网络中并没用 (用测试集)
        with torch.no_grad():   # TODO 看看,好像不太对,精确度和召回率同步了?
            P, R, mAP = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size, model=model)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (P, R, mAP) + '\n')
    return loss_list

# 迁移学习记得改cfg的class和卷积层输出3*(4+4+类)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/turtlebot.data', help='coco.data file path')  # TODO 改
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    loss_list = train(
                        opt.cfg,
                        opt.data_cfg,
                        img_size=opt.img_size,
                        resume=opt.resume,
                        epochs=opt.epochs,
                        batch_size=opt.batch_size,
                        accumulate=opt.accumulate,
                        multi_scale=opt.multi_scale,
                    )
