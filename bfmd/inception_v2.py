import torch

import torch.nn as nn

import torch.nn.functional as F



class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(

            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(out_planes, eps=.001)

        self.relu = nn.ReLU(inplace=False)



    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        return x





class Mixed_5b(nn.Module):

    def __init__(self):

        super(Mixed_5b, self).__init__()



        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)



        self.branch1 = nn.Sequential(

            BasicConv2d(192, 48, kernel_size=1, stride=1),

            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)

        )



        self.branch2 = nn.Sequential(

            BasicConv2d(192, 64, kernel_size=1, stride=1),

            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),

            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)

        )



        self.branch3 = nn.Sequential(

            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),

            BasicConv2d(192, 64, kernel_size=1, stride=1)

        )



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out





class Block35(nn.Module):

    def __init__(self, scale=1.0):

        super(Block35, self).__init__()



        self.scale = scale



        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)



        self.branch1 = nn.Sequential(

            BasicConv2d(320, 32, kernel_size=1, stride=1),

            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        )



        self.branch2 = nn.Sequential(

            BasicConv2d(320, 32, kernel_size=1, stride=1),

            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),

            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)

        )



        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=False)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)

        out = self.conv2d(out)

        out = out * self.scale + x

        out = self.relu(out)

        return out





class Mixed_6a(nn.Module):

    def __init__(self):

        super(Mixed_6a, self).__init__()



        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)



        self.branch1 = nn.Sequential(

            BasicConv2d(320, 256, kernel_size=1, stride=1),

            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),

            BasicConv2d(256, 384, kernel_size=3, stride=2)

        )



        self.branch2 = nn.MaxPool2d(3, stride=2)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)

        return out





class Block17(nn.Module):

    def __init__(self, scale=1.0):

        super(Block17, self).__init__()



        self.scale = scale



        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)



        self.branch1 = nn.Sequential(

            BasicConv2d(1088, 128, kernel_size=1, stride=1),

            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),

            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))

        )



        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=False)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        out = torch.cat((x0, x1), 1)

        out = self.conv2d(out)

        out = out * self.scale + x

        out = self.relu(out)

        return out





class Mixed_7a(nn.Module):

    def __init__(self):

        super(Mixed_7a, self).__init__()



        self.branch0 = nn.Sequential(

            BasicConv2d(1088, 256, kernel_size=1, stride=1),

            BasicConv2d(256, 384, kernel_size=3, stride=2)

        )



        self.branch1 = nn.Sequential(

            BasicConv2d(1088, 256, kernel_size=1, stride=1),

            BasicConv2d(256, 288, kernel_size=3, stride=2)

        )



        self.branch2 = nn.Sequential(

            BasicConv2d(1088, 256, kernel_size=1, stride=1),

            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),

            BasicConv2d(288, 320, kernel_size=3, stride=2)

        )



        self.branch3 = nn.MaxPool2d(3, stride=2)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out





class Block8(nn.Module):



    def __init__(self, scale=1.0, no_relu=False):

        super(Block8, self).__init__()



        self.scale = scale



        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)



        self.branch1 = nn.Sequential(

            BasicConv2d(2080, 192, kernel_size=1, stride=1),

            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),

            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        )



        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)

        self.relu = None if no_relu else nn.ReLU(inplace=False)



    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        out = torch.cat((x0, x1), 1)

        out = self.conv2d(out)

        out = out * self.scale + x

        if self.relu is not None:

            out = self.relu(out)

        return out





class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0., output_stride=32):

        super(InceptionResnetV2, self).__init__()

        self.drop_rate = drop_rate

        self.num_classes = num_classes

        self.num_features = 1536

        assert output_stride == 32



        self.conv2d_1a = BasicConv2d(in_chans, 32, kernel_size=3, stride=2)

        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)

        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.feature_info = [dict(num_chs=64, reduction=2, module='conv2d_2b')]



        self.maxpool_3a = nn.MaxPool2d(3, stride=2)

        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)

        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)

        self.feature_info += [dict(num_chs=192, reduction=4, module='conv2d_4a')]



        self.maxpool_5a = nn.MaxPool2d(3, stride=2)

        self.mixed_5b = Mixed_5b()

        self.repeat = nn.Sequential(

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17),

            Block35(scale=0.17)

        )

        self.feature_info += [dict(num_chs=320, reduction=8, module='repeat')]



        self.mixed_6a = Mixed_6a()

        self.repeat_1 = nn.Sequential(

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10),

            Block17(scale=0.10)

        )

        self.feature_info += [dict(num_chs=1088, reduction=16, module='repeat_1')]



        self.mixed_7a = Mixed_7a()

        self.repeat_2 = nn.Sequential(

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20),

            Block8(scale=0.20)

        )

        self.block8 = Block8(no_relu=True)

        self.conv2d_7b = BasicConv2d(2080, self.num_features, kernel_size=1, stride=1)

        self.feature_info += [dict(num_chs=self.num_features, reduction=32, module='conv2d_7b')]



        self.classif = nn.Linear(self.num_features, self.num_classes)



    def forward_features(self, x):

        x = self.conv2d_1a(x)

        x = self.conv2d_2a(x)

        x = self.conv2d_2b(x)

        x = self.maxpool_3a(x)

        x = self.conv2d_3b(x)

        x = self.conv2d_4a(x)

        x = self.maxpool_5a(x)

        x = self.mixed_5b(x)

        x = self.repeat(x)

        x = self.mixed_6a(x)

        x = self.repeat_1(x)

        x = self.mixed_7a(x)

        x = self.repeat_2(x)

        x = self.block8(x)

        x = self.conv2d_7b(x)

        return x



    def forward(self, x):

        x = self.forward_features(x)

        adaptiveAvgPoolWidth = x.shape[2]

        x = F.avg_pool2d(x, kernel_size=adaptiveAvgPoolWidth)

        if self.drop_rate > 0:

            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = x.view(x.size(0), -1)

        x = self.classif(x)

        return x





def inception_v2():

    return InceptionResnetV2(num_classes=1000)





if __name__ == '__main__':

    import torch

    image = torch.randn(2, 3, 416, 416)

    model = InceptionResnetV2(num_classes=1000)

    outputs = model(image)

    for output in outputs:

        print(output.shape)

    # print(prob.size())



    import io

    import numpy as np

    import torch.onnx

    model.eval()

    x = torch.randn(1, 3, 416, 416, requires_grad=True)



    from torchstat import stat

    stat(model,(3,416,416))
