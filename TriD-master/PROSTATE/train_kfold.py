# coding:utf-8
import torch
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
from torchnet import meter
from torch.autograd import Variable
from networks.ResUnet import ResUnet
from config import *
from utils.metrics import calculate_metrics
import numpy as np
import datetime


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


class Train:
    def __init__(self, config, train_loader, test_loader):
        # 数据加载
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 模型
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type

        # 损失函数
        self.seg_cost = Seg_loss()

        # 优化器
        self.optimizer = None
        self.scheduler = None
        self.optim = config.optimizer
        self.lr_scheduler = config.lr_scheduler
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.betas = (config.beta1, config.beta2)

        # 训练设置
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # 路径设置
        self.model_path = config.model_path
        self.result_path = config.result_path

        # 其他
        self.warm_up = -1
        self.valid_frequency = 10   # 多少轮测试一次
        self.device = config.device

        self.build_model()
        self.print_network()

    def build_model(self):
        if self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet='resnet34', num_classes=self.out_ch, pretrained=True).to(self.device)
        else:
            raise ValueError('The model type is wrong!')

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        self.optimizer = op_copy(self.optimizer)

        if torch.cuda.device_count() > 1:
            device_ids = list(range(0, torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        if self.lr_scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-7)
        elif self.lr_scheduler == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        elif self.lr_scheduler == 'Epoch':
            self.scheduler = EpochLR(self.optimizer, epochs=self.num_epochs, gamma=0.9)
        else:
            self.scheduler = None

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        best_loss, best_epoch = np.inf, 0
        loss_meter = meter.AverageValueMeter()
        # 绘制loss曲线
        metrics_l = {'Dice': [], 'ASD': []}
        metric_dict = ['Dice', 'ASD']

        for epoch in range(self.num_epochs):
            self.model.train()
            print("Epoch:{}/{}".format(epoch + 1, self.num_epochs))
            print("Training...")
            print("Learning rate: " + str(self.optimizer.param_groups[0]["lr"]))
            loss_meter.reset()
            metrics_y = [[], []]
            start_time = datetime.datetime.now()

            for batch, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                for para in self.model.parameters():
                    para.requires_grad = True

                CS_SS, y = data['data'], data['mask']
                CS_SS = torch.from_numpy(normalize_image(CS_SS)).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                CS_SS, y = CS_SS.to(self.device), y.to(self.device)

                pred = self.model(CS_SS)

                loss = self.seg_cost(pred, y)
                loss.backward()

                loss_meter.add(loss.sum().item())

                self.optimizer.step()

                # seg_output = torch.nn.Sigmoid()(pred.detach())
                # metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
                # for i in range(len(metrics)):
                #     metrics_y[i].append(metrics[i])

            if self.scheduler is not None:
                self.scheduler.step()

            # print_train = {}
            # train_metrics_y = np.sum(metrics_y, axis=1) / len(self.train_loader)
            #
            # for i in range(len(train_metrics_y)):
            #     metrics_l[metric_dict[i]].append(train_metrics_y[i])
            #     print_train[metric_dict[i]] = train_metrics_y[i]

            print("Train ———— Total_Loss:{:.8f}".format(loss_meter.value()[0]))
            # print("Train Metrics: ", print_train)

            if best_loss > loss_meter.value()[0]:
                best_loss = loss_meter.value()[0]
                best_epoch = (epoch + 1)

            # model_state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
            # torch.save(model_state, self.model_path + '/' + str(epoch+1) + '-' + self.model_type + '.pth')
                if torch.cuda.device_count() > 1:
                    torch.save(self.model.module.state_dict(), self.model_path + '/' + 'best' + '-' + self.model_type + '.pth')
                else:
                    torch.save(self.model.state_dict(), self.model_path + '/' + 'best' + '-' + self.model_type + '.pth')

            end_time = datetime.datetime.now()
            time_cost = end_time - start_time
            print('This epoch took {:6f} s'.format(time_cost.seconds + time_cost.microseconds / 1000000.))
            print("===" * 10)

        if torch.cuda.device_count() > 1:
            torch.save(self.model.module.state_dict(),
                       self.model_path + '/' + 'last' + '-' + self.model_type + '.pth')
        else:
            torch.save(self.model.state_dict(), self.model_path + '/' + 'last' + '-' + self.model_type + '.pth')
        print('The best total loss:{} epoch:{}'.format(best_loss, best_epoch))
        test_dice, test_asd = self.test()
        return test_dice, test_asd

    def test(self):
        metrics_y = [[], []]
        checkpoint = torch.load(self.model_path + '/' + 'best' + '-' + self.model_type + '.pth',
                                map_location=lambda storage, loc: storage.cuda(0))
        test_model = ResUnet(resnet='resnet34', num_classes=self.out_ch, pretrained=True).to(self.device)
        test_model.load_state_dict(checkpoint)
        test_model.eval()
        last_name = None

        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                x, y, path = data['data'], data['mask'], data['name']
                x = torch.from_numpy(x).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                current_name = path
                if last_name is None:
                    last_name = path

                x, y = Variable(x).to(self.device), Variable(y).to(self.device)

                seg_logit = test_model(x)
                seg_output = torch.nn.Sigmoid()(seg_logit)

                if current_name != last_name:
                    metrics = calculate_metrics(seg_output3D, y3D)

                    for i in range(len(metrics)):
                        metrics_y[i].append(metrics[i])

                    del seg_output3D
                    del y3D
                else:
                    try:
                        seg_output3D = torch.cat((seg_output.unsqueeze(2).detach().cpu(), seg_output3D), 2)
                        y3D = torch.cat((y.unsqueeze(2).detach().cpu(), y3D), 2)
                    except:
                        seg_output3D = seg_output.unsqueeze(2).detach().cpu()
                        y3D = y.unsqueeze(2).detach().cpu()
                last_name = current_name

        test_metrics_y = np.mean(metrics_y, axis=1)
        return test_metrics_y[0], test_metrics_y[1]

