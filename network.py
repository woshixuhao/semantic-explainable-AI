import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import torch


class feature_net(nn.Module):
    def __init__(self, model, dim, n_classes):
        super(feature_net, self).__init__()

        if model == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.feature = nn.Sequential(*list(vgg.children())[:-1])
            self.feature.add_module('global average', nn.AvgPool2d(7))
        elif model == 'inceptionv3':
            inception = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inception.children())[:-1])
            self.feature._modules.pop('13')
            self.feature.add_module('global average', nn.AvgPool2d(35))
        elif model == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            self.feature = nn.Sequential(*list(resnet.children())[:-1])
        elif model == 'Alexnet':
            alexnet = models.alexnet(pretrained=True)
            self.feature = nn.Sequential(*list(alexnet.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(dim, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x,MODEL='PCA'):
        # vgg = models.vgg19(pretrained=True)
        # test=nn.Sequential(*list(vgg.children())[:-1])
        # print(test(x.cpu()).shape)
        if MODEL=='PCA':
            x = self.feature(x)
            #print(x.shape)
            x_feature=x[0,:,0,0]#.cpu().data.numpy()
            #x_feature = x

            x = x.view(x.size(0), -1)
            #print(x.shape)
            x = self.classifier(x)
            #print(x.shape)
            return x,x_feature


        if MODEL=='Classfier':
            x = x.view(x.size(0), -1)
            x=self.classifier(x)
            return x


class ANN(nn.Module):
    def __init__(self,in_neuron,hidden_neuron,out_neuron):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(in_neuron,hidden_neuron)
        self.layer2 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer3 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer4 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer5 = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        x=self.layer1(x)
        x=torch.sin(x)
        x=self.layer2(x)
        x=torch.sin(x)
        x=self.layer3(x)
        x=torch.sin(x)
        x=self.layer4(x)
        x=torch.sin(x)
        x=self.layer5(x)
        return x

#自定义损失函数
class PINNLossFunc(nn.Module):
    def __init__(self,h_data_choose):
        super(PINNLossFunc,self).__init__()
        self.h_data=h_data_choose
        return

    def forward(self,prediction):
        f1=torch.pow((prediction-self.h_data),2).sum()
        MSE=f1
        return MSE