# coding:utf-8
import cv2
import torch
import numpy as np
from networks.ResUnet import ResUnet
from utils.metrics import calculate_metrics


class Test:
    def __init__(self, config, test_loader):
        # 数据加载
        self.test_loader = test_loader

        # 模型
        self.model = None
        self.model_type = config.model_type

        # 路径设置
        self.target = config.Target_Dataset
        self.result_path = config.result_path
        self.model_path = config.model_path

        # 其他
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.mode = config.mode
        self.device = config.device

        self.build_model()
        self.print_network()

    def build_model(self):
        if self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet='resnet34', num_classes=self.out_ch, pretrained=True,
                                 mixstyle_layers=[]).to(self.device)
        else:
            raise ValueError('The model type is wrong!')

        checkpoint = torch.load(self.model_path + '/' + 'best' + '-' + self.model_type + '.pth',
                                map_location=lambda storage, loc: storage.cuda(0))
        self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of parameters: {}".format(num_params))

    def test(self):
        print("Testing and Saving the results...")
        print("--" * 15)
        metrics_y = [[], []]
        metric_dict = ['Dice', 'ASD']

        last_name = None
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                x, y, path = data['data'], data['mask'], data['name']

                current_name = path
                if last_name is None:
                    last_name = path

                x = torch.from_numpy(x).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                x = x.to(self.device)
                seg_logit = self.model(x)
                seg_output = torch.sigmoid(seg_logit.detach().cpu())

                if current_name != last_name:   # Calculate the previous 3D volume
                    metrics = calculate_metrics(seg_output3D, y3D)
                    for i in range(len(metrics)):
                        metrics_y[i].append(metrics[i])

                    del seg_output3D
                    del y3D

                try:
                    seg_output3D = torch.cat((seg_output.unsqueeze(2), seg_output3D), 2)
                    y3D = torch.cat((y.unsqueeze(2), y3D), 2)
                except:
                    seg_output3D = seg_output.unsqueeze(2)
                    y3D = y.unsqueeze(2)

                draw_output = (seg_output.detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(self.result_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_pred.png', draw_output[0][0])
                last_name = current_name

        # Calculate the last 3D volume
        metrics = calculate_metrics(seg_output3D, y3D)
        for i in range(len(metrics)):
            metrics_y[i].append(metrics[i])

        test_metrics_y = np.mean(metrics_y, axis=1)
        print_test_metric = {}
        for i in range(len(test_metrics_y)):
            print_test_metric[metric_dict[i]] = test_metrics_y[i]

        with open('test_'+self.target+'.txt', 'w', encoding='utf-8') as f:
            f.write('Dice\n')
            f.write(str(metrics_y[0])+'\n')  # Dice
            f.write('ASD\n')
            f.write(str(metrics_y[1])+'\n')  # ASD

        print("Test Metrics: ", print_test_metric)
        return test_metrics_y
