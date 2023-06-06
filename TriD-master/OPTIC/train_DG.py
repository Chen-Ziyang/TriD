import torch
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
from torchnet import meter
from networks.ResUnet import ResUnet
from config import *
import numpy as np
from tensorboardX import SummaryWriter
from test import Test
import datetime


class TrainDG:
    def __init__(self, config, train_loader, valid_loader=None):
        # 数据加载
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # 模型
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type
        self.mixstyle_layers = config.mixstyle_layers
        self.random_type = config.random_type
        self.random_prob = config.random_prob

        # 损失函数
        self.seg_cost = Seg_loss()

        # 优化器
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
        self.log_path = config.log_path
        self.warm_up = -1
        self.valid_frequency = 5   # 多少轮测试一次
        self.device = config.device

        self.build_model()
        self.print_network()

    def build_model(self):
        if self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=True,
                                 mixstyle_layers=self.mixstyle_layers, random_type=self.random_type, p=self.random_prob).to(self.device)
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
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

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
        writer = SummaryWriter(self.log_path.replace('.log', '.writer'))
        best_loss, best_epoch = np.inf, 0
        loss_meter = meter.AverageValueMeter()

        metrics_l = {'Disc_Dice': [], 'Disc_ASD': [], 'Cup_Dice': [], 'Cup_ASD': []}
        metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']

        for epoch in range(self.num_epochs):
            self.model.train()
            print("Epoch:{}/{}".format(epoch + 1, self.num_epochs))
            print("Training...")
            print("Learning rate: " + str(self.optimizer.param_groups[0]["lr"]))
            loss_meter.reset()
            start_time = datetime.datetime.now()
            for batch, data in enumerate(self.train_loader):
                x, y = data['data'], data['mask']
                x = torch.from_numpy(normalize_image(x)).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)

                loss = self.seg_cost(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                loss_meter.add(loss.sum().item())

                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            writer.add_scalar('Total_Loss_Epoch', loss_meter.value()[0], epoch + 1)

            print("Train ———— Total_Loss:{:.8f}".format(loss_meter.value()[0]))

            if best_loss > loss_meter.value()[0]:
                best_loss = loss_meter.value()[0]
                best_epoch = (epoch + 1)

                if torch.cuda.device_count() > 1:
                    torch.save(self.model.module.state_dict(), self.model_path + '/' + 'best' + '-' + self.model_type + '.pth')
                else:
                    torch.save(self.model.state_dict(), self.model_path + '/' + 'best' + '-' + self.model_type + '.pth')

            if (epoch + 1) % self.valid_frequency == 0 and self.valid_loader is not None:
                test = Test(config=self.config, test_loader=self.valid_loader)
                result_dict = test.test()
                writer.add_scalar('Valid Disc Dice', result_dict['Disc_Dice'], (epoch + 1) // self.valid_frequency)
                writer.add_scalar('Valid Cup Dice', result_dict['Cup_Dice'], (epoch + 1) // self.valid_frequency)
                del test

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
        writer.close()

