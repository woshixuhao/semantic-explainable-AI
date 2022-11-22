import numpy as np
import scipy
from sklearn.decomposition import PCA
import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter
from network import feature_net
from collections import OrderedDict
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import cv2
import scipy.stats as st
from PIL import Image
from matplotlib import rc
font1 = {'family': 'Arial',
         'weight': 'normal',
         # "style": 'italic',
         'size': 7,
         }
font2 = {'family': 'Arial',
         'weight': 'normal',
         # "style": 'italic',
         'size': 9,
         }
legend_font={'family': 'Arial',
         'weight': 'normal',
         # "style": 'italic',
         'size': 6,
         }

# 参数设置
parser = argparse.ArgumentParser(description='cifar10')
parser.add_argument('--pre_epoch', default=0, help='begin epoch')
parser.add_argument('--total_epoch', default=1, help='time for ergodic')
parser.add_argument('--model', default='vgg', help='model for training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--pre_model', default=False, help='use pre-model')  # 恢复训练时的模型路径
args = parser.parse_args()

# 定义使用模型
model = args.model
# 使用gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor()])#,
                                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_seed', type=int, default=525, help='random seeds for torch')
    parser.add_argument('--path', type=str, default='D:/pycharm project/VAE_PDE/dogs_vs_cats/', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--fig_size', type=int, default=224, help='fig_size')


    parser.add_argument('--model', type=str, default='vgg', help='used model')
    parser.add_argument('--model_dim', type=int, default=512, help='used model dim, for vgg is 512')
    parser.add_argument('--TV_beta', type=int, default=2, help='beta for TV')

    parser.add_argument('--inverse_technique',type=str,default='GD',help='inverse methods including GD,unet')
    parser.add_argument('--layer_type', type=int, default=2, help='0,1,2')
    parser.add_argument('--visual_layer', type=int, default=0, help='visual_layer')
    parser.add_argument('--PCA_main', type=int, default=0, help='selected PCA main')
    parser.add_argument('--PCA_animal', type=str, default='dog', help='dog or cat')
    parser.add_argument('--dataset_situation', type=str, default='val', help='the used dataset train, val or train_dog')
    parser.add_argument('--PCA_data_num', type=int, default=500, help='selected PCA main')
    parser.add_argument('--L2_norm', type=float, default=0, help='selected PCA main')
    #----------params for GD--------------
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--TV_coef', type=float, default=2, help='beta for TV')
    parser.add_argument('--max_epoch', type=int, default=4000, help='max_epoch')

    parser.add_argument('--visual_layer_start', type=int, default=0, help='start_layer')
    parser.add_argument('--visual_layer_end', type=int, default=1, help='end_layer')
    parser.add_argument('--plot_row', type=int, default=1, help='plot_row')
    parser.add_argument('--plot_col', type=int, default=2, help='plot_col')

    #------------------PCA position-------------
    parser.add_argument('--position_animal', type=str, default='cat', help='position animal, cat or dog')
    parser.add_argument('--position_target', type=str, default='eye', help='position animal, cat or dog')
    parser.add_argument('--position_space', type=str, default='eye', help='position animal, cat or dog')
    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.std=[0.5,0.5,0.5]
    config.mean = [0.5, 0.5, 0.5]
    return config

class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        try:
            self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        except TypeError:
            self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


config= parse_args()


def generate_dataset():
    '''
    load dataset
    :return: dataset
    '''
    torch.manual_seed(config.torch_seed)
    transform = transforms.Compose([transforms.CenterCrop(config.fig_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    data_image = {x: datasets.ImageFolder(root=os.path.join(config.path, x),
                                          transform=transform)
                  for x in ["train", "val"]}
    data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                        batch_size=config.batch_size,
                                                        shuffle=False)
                         for x in ["train", "val"]}

    data_image_dog = datasets.ImageFolder(root=os.path.join(config.path, 'train_dog'),
                                          transform=transform)

    data_loader_dog = torch.utils.data.DataLoader(dataset=data_image_dog,
                                                  batch_size=config.batch_size,
                                                  shuffle=False)

    if config.dataset_situation == 'train_dog':
        return data_loader_dog
    if config.dataset_situation == 'train':
        return data_loader_image['train']
    if config.dataset_situation == 'val':
        return data_loader_image['val']

def recover_to_image(image_origin):
    '''
    standardize the image
    :param image_origin:
    :return: standardized image
    '''
    img = image_origin.copy()
    img = (img - config.mean) / config.std
    img = img.transpose((2, 0, 1))
    image = torch.from_numpy(img.astype(np.float32)).reshape([1, 3, 224, 224])
    return image

def transform_raw_picture(pic):
    '''
    unstandardize the image
    :param standardized image
    :return: standardized image
    '''
    img = torchvision.utils.make_grid(pic)  # 把batch_size张的图片拼成一个图片
    img = img.numpy().transpose((1, 2, 0))  # 本来是(0,1,2)，相当于把第一维变为第三维，其他两维前移
    img = img * config.std + config.mean  # (228, 906, 3)范围由(-1, 1)变成(0, 1)
    return img

def generate_feature_matrix(seed,lime=True):
    '''
    extract the feature maps
    :param seed: random seeds
    :param lime: whether use GA
    :return: save the extracted feature maps
    '''
    use_model = feature_net(config.model, dim=config.model_dim, n_classes=2)  # for vgg dim=512
    use_model.load_state_dict(torch.load(f"CNN_model_save/VGG_{config.fig_size}.pth"))
    use_model.eval()
    use_model.to(config.device)
    feature_save = []
    conv_out = LayerActivations(use_model.feature[config.layer_type], config.visual_layer) #feture maps after GAP
    image_best_save = torch.from_numpy(np.load(f'lime_save/lime_{config.PCA_animal}_500_{seed}.npy').astype(np.float32))
    image_origin_save = torch.from_numpy(np.load(f'lime_save/lime_{config.PCA_animal}_500_origin_{seed}.npy').astype(np.float32))

    for i in range(image_best_save.shape[0]):
        print(f'--------第{i}次处理-------')
        if lime==True:
            image= image_best_save[i].reshape([1,3,224,224])
        else:
            image = image_origin_save[i].reshape([1, 3, 224, 224])
        image= Variable(image.to(config.device))
        _, _ = use_model(image)
        x_feature = conv_out.features
        feature_save.append(x_feature.data.numpy())
        if i == config.PCA_data_num - 1:
            break

    try:
        os.makedirs(f'feature_save_{config.PCA_animal}')
    except OSError:
        pass
    np.save("feature_save_%s/feature_matrix_%d_layer_%d_%d.npy" % (config.PCA_animal, config.PCA_data_num,config.layer_type, config.visual_layer),
            feature_save, allow_pickle=True)

def load_PCA(n_components = 10):
    '''
    row-centered PCA
    :param n_components: the number of saved components
    :return: obtained PC after relu
    '''
    feature_matrix = np.load(
        "feature_save_%s/feature_matrix_%d_layer_%d_%d.npy" % (config.PCA_animal, config.PCA_data_num,config.layer_type, config.visual_layer))
    feature_matrix_main = np.zeros(
        [feature_matrix.shape[1], feature_matrix.shape[2], feature_matrix.shape[3], feature_matrix.shape[4]])


    ratio_record = []

    feature_matrix_total = np.zeros(
        [n_components, feature_matrix.shape[2], feature_matrix.shape[3], feature_matrix.shape[4]])

    for i in range(feature_matrix.shape[3]):
        for j in range(feature_matrix.shape[4]):
            feature_matrix_PCA = np.squeeze(feature_matrix[:, 0, :, i, j]).T
            pca = PCA(n_components=n_components)
            pca.fit(feature_matrix_PCA)
            ratio_record.append(pca.explained_variance_ratio_[0])
            feature_space = pca.fit_transform(feature_matrix_PCA)
            feature_matrix_main[0, :, i, j] = feature_space[:, config.PCA_main]
            feature_matrix_total[:, :, i, j] = feature_space[:, 0:n_components].T
            print(i, j)

    return torch.nn.functional.relu(torch.from_numpy(feature_matrix_main.astype(np.float32)))

def total_variance(img, beta=2):
    '''
    TV loss
    :param img:
    :param beta:
    :return: TC loss
    '''
    TV_row = torch.clone(img)
    TV_col = torch.clone(img)
    _, C, H, W = img.shape
    TV_row[:, :, :, 0:W - 1] = img[:, :, :, 1:W] - img[:, :, :, 0:W - 1]
    TV_col[:, :, 0:H - 1, :] = img[:, :, 1:H, :] - img[:, :, 0:H - 1, :]
    TV = ((TV_row ** 2 + TV_col ** 2) ** (beta / 2)).sum()


    return TV

def picture_inverse(use_model,feature_true,max_epoch, TV=True):
    '''
    visualization of common traits or semantic space
    :param use_model: trained CNN
    :param feature_true: target feature to be visualized
    :param max_epoch: max epoch
    :param TV: whether use TV loss
    :return: visualization picture
    '''
    if config.inverse_technique=='GD':
        learning_rate=config.learning_rate
        # -------产生空白图----------
        pic_prior = torch.zeros([1, 3, config.fig_size, config.fig_size])

        pic_prior = Variable(pic_prior.to(config.device), requires_grad=True)
        feature_true = Variable(feature_true.to(config.device))

        conv_out = LayerActivations(use_model.feature[config.layer_type], config.visual_layer)

        _, feature_prediction = use_model(pic_prior)
        act = conv_out.features
        feature_prediction = act.to(config.device)

        if TV == True:
            MSE = torch.sum((feature_prediction - feature_true) ** 2) +config.TV_coef*total_variance(pic_prior, beta=config.TV_beta)
        else:
            MSE = torch.sum((feature_prediction - feature_true) ** 2)


        # -----迭代下降
        H_grad = torch.autograd.grad(outputs=MSE.sum(), inputs=pic_prior, create_graph=True)[0]
        mu = - learning_rate * H_grad
        pic_new = pic_prior - learning_rate * H_grad

        img = torchvision.utils.make_grid(pic_new)  # 把batch_size张的图片拼成一个图片
        img = img.cpu().data.numpy().transpose((1, 2, 0))  # 本来是(0,1,2)，相当于把第一维变为第三维，其他两维前移
        img = img * config.std + config.mean  # (228, 906, 3)范围由(-1, 1)变成(0, 1)
        # plt.imshow(img)
        # plt.show()

        # inv_pic_plot = Visualization.transform_raw_picture(config,pic_new.cpu().data)
        # plt.imshow(inv_pic_plot)
        # plt.show()

        for epoch in range(max_epoch):
            a=time.time()
            _, feature_prediction = use_model(pic_new)
            act = conv_out.features
            feature_prediction = act.to(config.device)
            if TV == True:
                MSE = torch.sum((feature_prediction - feature_true) ** 2) +config.TV_coef *total_variance(pic_new, beta=config.TV_beta)+config.L2_norm*torch.mean(pic_new**2)
            else:
                MSE = torch.sum((feature_prediction - feature_true) ** 2)
            H_grad = torch.autograd.grad(outputs=MSE.sum(), inputs=pic_new, create_graph=True)[0]
            pic_new = (pic_new - learning_rate * H_grad).cpu().data.numpy()


            pic_new = Variable(torch.from_numpy(pic_new.astype(np.float32)).to(config.device), requires_grad=True)
            b=time.time()
            #print(b-a,epoch, MSE, Visualization.total_variance(config,pic_new, beta=2))
            if (epoch + 1) % 100 == 0:
                print(f'mean:{torch.mean(pic_new**2)}')
                print(f'epoch:{epoch+1},MSE:{torch.sum((feature_prediction - feature_true) ** 2) /(feature_true.shape[0]*feature_true.shape[1]*feature_true.shape[2]*feature_true.shape[3])},'
                      f'TV:{config.TV_coef* total_variance(pic_new, beta=config.TV_beta)/(feature_true.shape[0]*feature_true.shape[1]*feature_true.shape[2]*feature_true.shape[3])}')
                # inv_pic_plot = Visualization.transform_raw_picture(config,pic_new.cpu().data)
                # plt.imshow(inv_pic_plot)
                # plt.show()

            if (epoch + 1) % 1000==0:
                learning_rate = learning_rate*0.5

    return pic_new.cpu().data, feature_prediction.cpu().data.numpy()

def cut_position(use_model):
    '''
    mask the semantic concept
    :param use_model: the trained CNN
    :return: images with masked and unmasked semantic concept
    '''
    n_segments =20
    sigma = 5
    segment_save=[]
    origin_save=[]
    config.dataset_situation='train_dog'
    for i, data in enumerate(generate_dataset()):
        print('----------第%d次迭代----------' % (i))
        image, label = data
        img = torchvision.utils.make_grid(image)
        img = img.cpu().data.numpy().transpose((1, 2, 0))  # 本来是(0,1,2)，相当于把第一维变为第三维，其他两维前移
        img = img * config.std + config.mean  # (228, 906, 3)范围由(-1, 1)变成(0, 1)

        # 查看分割图
        segments = slic(img_as_float(img), n_segments=n_segments, sigma=sigma)
        fig1=plt.figure(1)
        plt.imshow(mark_boundaries(img, segments))
        #Here, for each picture, we click 4 times on the position segment (e.g., eye, noise), note that the click position should be different.
        #if there is no position, we click 4 on the same position to skip this picture. 
        position_spot = plt.ginput(4, timeout=90)
        plt.close(fig1)
        if position_spot[0]==position_spot[1]:
            print('no position')
            continue
        print([position_spot])

        fig1 = plt.figure(1)
        plt.imshow(mark_boundaries(img, segments))
        color_spot = plt.ginput(1, timeout=90)
        plt.close(fig1)
        segment_index=[]
        color_index=0
        #segval_select = [2, 3, 4, 5, 6, 7, 8, 9, 10, 23, 45]


        for k in range(n_segments):
            mask = np.zeros(img.shape[:2], dtype="uint8")
            mask[segments == k] = 255
            if mask[int(color_spot[0][1]),int(color_spot[0][0])].sum()!=0:
                color_index=k

            for j in range(len(position_spot)):
                if mask[int(position_spot[j][1]),int(position_spot[j][0])].sum()!=0:
                    segment_index.append(k)


        segment_index=np.array(segment_index)
        segment_index=np.unique(segment_index)
        segment_index=list(segment_index)

        mask_color=np.zeros(img.shape[:2], dtype="uint8")
        mask_color[segments == color_index] = 255
        color_split = np.multiply(img, cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR) > 0)
        print(color_split.shape)
        average_color=np.zeros([3])
        for i in range(3):
            average_color[i]=np.sum(color_split[:,:,i])/np.count_nonzero(color_split[:,:,i])
        print(average_color)



        mask = np.zeros(img.shape[:2], dtype="uint8")
        for k in range(len(segment_index)):
            mask[segments == segment_index[k]] = 255
        image_split = np.multiply(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0)
        image_color=image_split.copy()
        for i in range(224):
            for j in range(224):
                if image_color[i,j].sum()!=0:
                    image_color[i,j]=average_color

        image_revise=img-image_split+image_color
        image_save=recover_to_image(image_revise).data.numpy()
        segment_save.append(image_save)
        origin_save.append(image.cpu().data.numpy())
        fig2=plt.figure(2)
        # plt.imshow()
        # plt.draw()
        # plt.show()
        plt.close(fig1)
        # plt.pause(1)
        # plt.close(fig2)

        if len(segment_save)==100:
            segment_save=np.array(segment_save)
            origin_save=np.array(origin_save)
            np.save("result_save/leg_save_dog.npy", segment_save)
            np.save("result_save/leg_save_dog_origin.npy", origin_save)
            break

def position_PCA(feature_matrix,n_components):
    '''
    :param feature_matrix: the obtained feature matrix for semantic space
    :param n_components: preserve components
    :return: PCs
    '''
    n_components_record = []
    ratio_record = []
    ratio_record_origin = []

    feature_matrix_main = np.zeros(
        [feature_matrix.shape[1], feature_matrix.shape[2], feature_matrix.shape[3], feature_matrix.shape[4]])

    feature_matrix_total = np.zeros(
        [n_components, feature_matrix.shape[2], feature_matrix.shape[3], feature_matrix.shape[4]])

    for i in range(feature_matrix.shape[3]):
        for j in range(feature_matrix.shape[4]):
            feature_matrix_PCA = np.squeeze(feature_matrix[:, 0, :, i, j]).T
            pca = PCA(n_components=n_components)
            pca.fit(feature_matrix_PCA)
            ratio_record.append(pca.explained_variance_ratio_[0])
            feature_space = pca.fit_transform(feature_matrix_PCA)
            feature_matrix_main[0, :, i, j] = feature_space[:, 0]
            feature_matrix_total[:, :, i, j] = feature_space[:, 0:n_components].T

    feature_matrix_main_1 = torch.nn.functional.relu(torch.from_numpy(feature_matrix_main.astype(np.float32)))
    return feature_matrix_main_1



def get_position(conv_out,N_select_PCA,N_select = 512,show_picture=False):
    '''
    conduct row-centered PCA and select the SSN
    :param conv_out: the layer
    :param N_select_PCA: the nunber of SSN
    :param N_select: the number of channels
    :param show_picture: whether show the picture
    '''
    position = torch.from_numpy(np.load(f'result_save/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))
    position_origin=torch.from_numpy(np.load(f'result_save/{config.position_space}_save_{config.position_animal}_origin.npy').astype(np.float32))
    if show_picture==True:
        for i in range(position.shape[0]):
            plt.figure(1)
            image_plot = transform_raw_picture(position[i].cpu().data)
            plt.imshow(image_plot)


    feature_save=[]
    feature_save_origin=[]
    for i in range(position.shape[0]):
        image = position[i]
        image_origin=position_origin[i]


        label_true, x_feature = use_model(image.cuda())
        act = conv_out.features
        x_feature = act.to(device).reshape(512 )

        feature_save.append(act.data.numpy())

        label_true, x_feature = use_model(image_origin.cuda())
        act = conv_out.features
        feature_save_origin.append(act.data.numpy())
        x_feature_origin = act.to(device).reshape(512)

        if show_picture == True:
            plt.figure(2)
            plt.subplot(10,10,i+1)
            plt.plot((x_feature).cpu().data.numpy())
            plt.plot((-x_feature_origin).cpu().data.numpy())


    feature_matrix=np.array(feature_save)
    feature_matrix_origin = np.array(feature_save_origin)
    feature_PCA=position_PCA(feature_matrix,n_components=10)
    feature_PCA_origin=position_PCA(feature_matrix_origin,n_components=10)

    if show_picture==True:
        plt.figure(3,figsize=(2,2),dpi=300)
        plt.bar(range(512), feature_PCA_origin[0, :, 0, 0].cpu().data.numpy(), color='red', label=f'With {config.position_space}s')
        plt.bar(range(512), -feature_PCA[0,:,0,0].cpu().data.numpy(), color='blue',label=f'With {config.position_space}s masked')
        plt.xlabel('Neuron Index', fontproperties='Arial', fontsize=7)
        plt.ylabel('Scores of $1^{st}$ PC', fontproperties='Arial', fontsize=7)
        plt.ylim(-20, 30)
        plt.xticks(fontproperties='Arial', size=7)
        plt.yticks([ -20,-10,  0, 10, 20,30], [20, 10, 0, 10, 20,30], fontproperties='Arial', size=7)
        plt.legend(loc='lower right',  # 图例的底部中央位置在图像上部居中
    frameon=False,  # 不显示图例框线
    prop=legend_font)
        # plt.savefig(f'PPT_fig/position/{config.position_space}_{config.position_animal}.tiff',
        #             bbox_inches='tight', dpi=300)
        # plt.savefig(f'PPT_fig/position/{config.position_space}_{config.position_animal}.pdf',
        #             bbox_inches='tight', dpi=300)


        plt.figure(4, figsize=(2, 2), dpi=300)
        sort_index_cat = np.argsort(-np.abs(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy()))[0:1]  # 从小到大
        plt.bar(range(512),-(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy()))
        for i in range(len(sort_index_cat)):
            if sort_index_cat[i]<0:
                plt.text(sort_index_cat[i], -(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy())[sort_index_cat[i]] + 0.25, '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
                     fontproperties='Arial', fontsize=7)
            else:
                plt.text(sort_index_cat[i], -(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy())[sort_index_cat[i]] - 0.25,
                         '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
                         fontproperties='Arial', fontsize=7)
        plt.xlabel('Neuron Index', fontproperties='Arial', fontsize=7)
        plt.ylabel('Difference', fontproperties='Arial', fontsize=7)
        plt.xticks(fontproperties='Arial', size=7)
        plt.yticks( fontproperties='Arial',size=7)
        plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}.tiff',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}.pdf',
                    bbox_inches='tight', dpi=300)

    sort_index_origin = np.argsort(-(feature_PCA_origin[0,:,0,0]))[0:N_select]
    PCA_cat=(feature_PCA_origin[0, :, 0, 0]-feature_PCA[0,:,0,0])[sort_index_origin].cpu().data.numpy()
    sort_index_position=np.argsort(-(np.abs(PCA_cat)))[0:N_select_PCA]
    #print(np.sign(PCA_cat[sort_index_position]))
    space_value=PCA_cat[sort_index_position]
    #print(sort_index_origin[sort_index_position]*np.sign(PCA_cat[sort_index_position]))
    space_index=sort_index_origin[sort_index_position] * np.sign(PCA_cat[sort_index_position])
    if show_picture==True:
        plt.show()
    return sort_index_origin,sort_index_position,space_index,space_value



def get_space(conv_out, reshape_num,add_position='origin'):
    '''
    get the feature maps of images with masked and unmasked semantic concepts
    :param conv_out:
    :param reshape_num:
    :param add_position:
    :return:
    '''
    cat_space = []
    cat_space_pic=[]
    if add_position=='origin':
        position = torch.from_numpy(np.load(f'result_save/{config.position_target}_save_{config.position_animal}.npy').astype(np.float32))
        position_origin = torch.from_numpy(np.load(f'result_save/{config.position_target}_save_{config.position_animal}_origin.npy').astype(np.float32))

        for i in range(100):
            print('----------第%d次迭代----------' % (i))
            image_origin = position_origin[i]
            image = image_origin
            image = Variable(image.to(device), requires_grad=True)
            label_true, x_feature = use_model(image)
            act = conv_out.features
            x_feature = act.to(device).reshape(reshape_num)
            cat_space.append(x_feature.cpu().data.numpy())
            cat_space_pic.append(image.cpu().data.numpy())

    elif add_position=='position':
        position = torch.from_numpy(np.load(f'result_save/{config.position_target}_save_{config.position_animal}.npy').astype(np.float32))
        position_origin = torch.from_numpy(np.load(f'result_save/{config.position_target}_save_{config.position_animal}_origin.npy').astype(np.float32))

        for i in range(100):
            print('----------第%d次迭代----------' % (i))
            image_origin = position[i]
            image = image_origin
            image = Variable(image.to(device), requires_grad=True)
            label_true, x_feature = use_model(image)
            act = conv_out.features
            x_feature = act.to(device).reshape(reshape_num)
            cat_space.append(x_feature.cpu().data.numpy())
            cat_space_pic.append(image.cpu().data.numpy())

    cat_space=np.array(cat_space)
    cat_space_pic=np.array(cat_space_pic)
    return cat_space,cat_space_pic


def val_distribution(conv_out, reshape_num):
    '''
    get featurem maps of 3000 samples
    :param conv_out:
    :param reshape_num:
    :return:
    '''
    cat_space = []
    cat_space_pic=[]
    config.dataset_situation='train_dog'
    for i, data in enumerate(generate_dataset()):
        print('----------第%d次迭代----------' % (i))
        if i<1000:
            continue
        image, label = data
        image = Variable(image.to(device), requires_grad=True)
        label_true, x_feature = use_model(image)
        act = conv_out.features
        x_feature = act.to(device).reshape(reshape_num).cpu().data.numpy()
        cat_space.append(x_feature)
        cat_space_pic.append(image.cpu().data.numpy())
        if i==2999:
            break

    cat_space = np.array(cat_space)
    cat_space_pic=np.array(cat_space_pic)
    print(cat_space.shape)
    print(cat_space_pic.shape)

    np.save('result_save/dog_space_2000.npy', cat_space)
    np.save('result_save/dog_space_2000_pic.npy', cat_space_pic)
    return cat_space,cat_space_pic



def get_cdf_ratio(x,mean):
    '''
    calculate the semantic probability
    :param x:
    :param mean:
    :return:
    '''
    min=st.norm.cdf((np.min(mean) - np.mean(mean)) / np.std(mean))
    max=st.norm.cdf((np.max(mean) - np.mean(mean)) / np.std(mean))
    return (st.norm.cdf((x - np.mean(mean)) / np.std(mean))-min)/(max-min)

def plot_distribution(space_index,space_value,picture='big'):
    '''
    calculate the mean and std of the fitted normal distribution
    :param space_index:
    :param space_value:
    :param picture:
    :return:
    '''

    space_index=np.array(space_index, dtype=int)
    space = np.load(f'result_save/{config.position_animal}_space_2000.npy')
    space_pic = np.load(f'result_save/{config.position_animal}_space_2000_pic.npy')
    eye_space = space[:, space_index]
    sum=0
    for j in range(eye_space.shape[1]):
        sum+=eye_space[:,j]*space_value[j]
    mean=sum/eye_space.shape[1]

    print(mean)
    import statsmodels.api as sm
    import pylab
    # sm.qqplot(mean, line='s')
    # pylab.show()

    from scipy import stats
    plt.figure(1, figsize=(3, 3), dpi=300)
    a,b=stats.probplot(mean, dist="norm",fit=True)
    print(b)
    plt.scatter(a[0],a[1],c='blue',s=9)
    plt.plot(a[0],b[0]*a[0]+b[1],color='red')
    plt.ylabel('Ordered values',font2)
    plt.xlabel('Theoretical quantities',font2)
    plt.xticks([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3],fontproperties='Arial', size=9)
    plt.yticks(fontproperties='Arial', size=9)
    plt.title(' ')
    plt.savefig(f'PPT_fig/position/qq_{config.position_animal}_{config.position_space}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/position/qq_{config.position_animal}_{config.position_space}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

    plt.figure(2,figsize=(2,2),dpi=300)
    #plt.figure(1,figsize=(6.5,1.5),dpi=300)
    mu = np.mean(mean)  # 均值μ
    sig = np.std(mean)  # 标准差δ

    x = np.linspace(np.min(mean),np.max(mean),100)  # 定义域
    y = np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)  # 定义曲线函数
    #plt.xlim(0,16)
    plt.plot(x, y, "red", linewidth=2)
    plt.hist(mean,bins=40,color='blue',density=True)
    plt.ylabel('Probability density',font1)
    plt.xlabel('Values of $A_s$',font1)
    plt.xticks(fontproperties='Arial', size=7)
    plt.yticks(fontproperties='Arial', size=7)

    plt.savefig(f'PPT_fig/position/dis_{config.position_animal}_{config.position_space}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/position/dis_{config.position_animal}_{config.position_space}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()



    plt.style.use('default')
    if picture=='big':
        for i in range(mean.shape[0]):
            if get_cdf_ratio(mean[i],mean) >0.9:
                print(get_cdf_ratio(mean[i],mean))
                image_origin_plot = transform_raw_picture(torch.from_numpy(space_pic[i]))
                plt.figure(2,figsize=(4,4),dpi=300)
                plt.imshow(image_origin_plot)
                plt.axis('off')
                # plt.savefig(f'PPT_fig/position/big_{config.position_animal}_{config.position_space}_{i}.tiff',
                #             bbox_inches='tight', dpi=300)
                # plt.savefig(f'PPT_fig/position/big_{config.position_animal}_{config.position_space}_{i}.pdf',
                #             bbox_inches='tight', dpi=300)
                plt.show()
    if picture=='small':
        for i in range(mean.shape[0]):
            if get_cdf_ratio(mean[i],mean) <0.1:
                print(get_cdf_ratio(mean[i],mean))
                image_origin_plot = transform_raw_picture(torch.from_numpy(space_pic[i]))
                plt.figure(2, figsize=(4, 4), dpi=300)
                plt.imshow(image_origin_plot)
                plt.axis('off')
                # plt.savefig(f'PPT_fig/position/small_{config.position_animal}_{config.position_space}_{i}.tiff',
                #             bbox_inches='tight', dpi=300)
                # plt.savefig(f'PPT_fig/position/small_{config.position_animal}_{config.position_space}_{i}.pdf',
                #             bbox_inches='tight', dpi=300)
                plt.show()


    if picture=='location':
        cat_space,cat_space_pic=get_space(conv_out_2_0,512,add_position='position')
        cat_space_origin, cat_space_pic_origin = get_space(conv_out_2_0, 512, add_position='origin')
        eye_space = cat_space[:, space_index]
        sum = 0
        for j in range(eye_space.shape[1]):
            sum += eye_space[:, j] * space_value[j]
        mean_location = sum / eye_space.shape[1]

        eye_space = cat_space_origin[:, space_index]
        sum = 0
        for j in range(eye_space.shape[1]):
            sum += eye_space[:, j] * space_value[j]
        mean_origin = sum / eye_space.shape[1]
        for i in range(mean_location.shape[0]):
            print(get_cdf_ratio(mean_location[i],mean))
            print(get_cdf_ratio(mean_origin[i],mean))
            plt.figure(6,dpi=300)
            plt.subplot(1,2,1)
            plt.imshow(transform_raw_picture(torch.from_numpy(cat_space_pic[i])))
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(transform_raw_picture(torch.from_numpy(cat_space_pic_origin[i])))
            plt.axis('off')
            # plt.savefig(f'PPT_fig/position/compare_{config.position_animal}_{config.position_space}_{i}.tiff',
            #             bbox_inches='tight', dpi=300)
            # plt.savefig(f'PPT_fig/position/compare_{config.position_animal}_{config.position_space}_{i}.pdf',
            #             bbox_inches='tight', dpi=300)
            plt.show()
            print('-------------------------')


def transform_pro(prediction):
    '''
    get the probability give by CNN
    :param prediction:
    :return:
    '''
    prediction=torch.flatten(prediction)
    prediction_cat=torch.exp(prediction[0])/(torch.exp(prediction[0])+torch.exp(prediction[1]))
    prediction_dog = torch.exp(prediction[1]) / (torch.exp(prediction[0]) + torch.exp(prediction[1]))
    return prediction_cat,prediction_dog

def plot_radian(data,name):
    '''
    plot radian plot
    :param data:
    :param name:
    :return:
    '''
    plt.style.use('ggplot')

    # data
    labels = np.array(['       dog nose', 'dog eye', 'cat eye', 'cat nose      ', 'cat leg', 'dog leg'])
    dataLenth = 6
    angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
    data = np.concatenate((data, [data[0]]))  # 闭合
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    # plot
    plt.rcParams['font.size'] = 5
    plt.rcParams['font.family']='Arial'
    plt.rcParams['font.style']='italic'
    #plt.rcParams['font.weight'] = 'bold'
    fig = plt.figure(figsize=(2, 2),dpi=300)
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, data, '--',color='r', linewidth=1)  # 画线
    ax.scatter(angles,data,c='r',s=15)
    ax.fill(angles, data, facecolor='r', alpha=0.35)  # 填充
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="Arial", fontsize=10,fontstyle='normal')
    ax.set_rlim(0,1)
    # ax.set_title("matplotlib雷达图", va='bottom', fontproperties="SimHei",fontsize=22)
    ax.grid(True)
    plt.savefig(f'PPT_fig/radian_{name}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/radian_{name}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

def print_word(conv_out,reshape_num):
    '''
    give the explanation and the trustworthiness assessment
    :param conv_out:
    :param reshape_num:
    :return:
    '''
    config.dataset_situation = 'val'
    # adverse_sample = torch.from_numpy(np.load(r'D:\pycharm project\VAE_PDE\fake_picture\cat-fake-dog-PGD-0.05.npy').astype(np.float32))
    # true_sample = torch.from_numpy(np.load(r'D:\pycharm project\VAE_PDE\fake_picture\cat-true-dog-PGD-0.01.npy').astype(np.float32))
    for i, data in enumerate(generate_dataset()):
        print('----------第%d次迭代----------' % (i))
        image, label = data
        #image=true_sample[i]
        transform = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        #fname = r'D:\pycharm project\VAE_PDE\big_cat_1.jpg'
        file_name='val_2'
        # image = Image.open(fname)
        # image = transform(image).reshape([1, 3, 224, 224])


        image = Variable(image.to(device), requires_grad=True)
        label_true, x_feature = use_model(image)

        print(transform_pro(label_true))
        act = conv_out.features
        x_feature = act.to(device).reshape(reshape_num)
        feature_space = x_feature
        pic = image.cpu().data
        img = transform_raw_picture(pic)
        plt.imshow(img)
        plt.axis('off')
    # if i==6:
    #         pic=image.cpu().data
    #         img=transform_raw_picture(pic)
    #         plt.imshow(img)
    #         break



        cdf_record=[]
        for i in ['eye','nose','leg']:
            for j in ['dog','cat']:
                config.position_space =i
                config.position_animal = j
                _, _, space_index, space_value = get_position(conv_out_2_0, 5, show_picture=False)
                space_index = np.array(space_index, dtype=int)
                #This file can be generated by running the function val_distribution with modifying dataset_situation and save_name
                space = np.load(f'result_save/{config.position_animal}_space_2000.npy')

                eye_space = space[:, space_index]
                sum = 0
                for k in range(eye_space.shape[1]):
                    sum += eye_space[:, k] * space_value[k]
                mean = sum / eye_space.shape[1]

                eye_space = feature_space[space_index].cpu().data.numpy()
                sum = 0
                for k in range(eye_space.shape[0]):
                    sum += eye_space[k] * space_value[k]
                mean_pic = sum / eye_space.shape[0]
                cdf_record.append(get_cdf_ratio(mean_pic,mean))
                print(f'{j},{i},{get_cdf_ratio(mean_pic,mean)}')
        dog_cdf=np.array(cdf_record)[[0,2,4]]
        cat_cdf=np.array(cdf_record)[[1,3,5]]
        delta=dog_cdf-cat_cdf
        word=''
        word_1=word_2=word_3=word_4=word_5=''
        word_list=['eyes','nose','legs']
        is_list=['are','is','are']
        if label_true[0,1]>=label_true[0,0]:
            if np.max(delta)<0.2:
                word+='It might be a dog, but I am not sure.'
            else:
                if np.max(delta)>0.5:
                    word+='I am sure it is a dog mainly because '
                else:
                    word+='It is probably a dog mainly because '
                for m in range(len(word_list)):
                    if dog_cdf[m] > 0.5 and delta[m] > 0.5:
                        if word_1=='':
                            word_1+=word_list[m]
                        else:
                            word_1 +=' and '
                            word_1+=word_list[m]
                    if dog_cdf[m] > 0.5 and 0.35 < delta[m] <= 0.5:
                        if word_2 == '':
                            word_2 += word_list[m]
                        else:
                            word_2+= ' and '
                            word_2 += word_list[m]
                    if dog_cdf[m] > 0.5 and 0.2 < delta[m] <= 0.35:
                        if word_3 == '':
                            word_3 += word_list[m]
                        else:
                            word_3 += ' and '
                            word_3 += word_list[m]
                    if 0.2 < dog_cdf[m] < 0.5 and 0.35 < delta[m] <= 0.5:
                        if word_4 == '':
                            word_4 += word_list[m]
                        else:
                            word_4 += ' and '
                            word_4 += word_list[m]
                    if 0.2 < dog_cdf[m] < 0.5 and 0.2 < delta[m] <= 0.35:
                        if word_5 == '':
                            word_5 += word_list[m]
                        else:
                            word_5 += ' and '
                            word_5 += word_list[m]
                print(word_1,word_2,word_3,word_4,word_5)
                #m=np.where(delta==np.max(delta))[0][0]
                flag=0
                if word_1!='':
                    is_are='are'
                    if word_1=='nose':
                        is_are='is'
                    word+=f"it has vivid {word_1}, which {is_are} dog's {word_1} obviously. "
                    flag=1
                if word_2 != '':
                    is_are = 'are'
                    if word_2 == 'nose':
                        is_are = 'is'
                    if flag==1:
                        word+=f"Meanwhile, it has vivid {word_2}, which {is_are} something like dog's {word_2}. "
                    if flag==0:
                        word += f"it has vivid {word_2}, which {is_are} something like dog's {word_2}. "

                if word_3 != '':
                    is_are = 'are'
                    if word_3 == 'nose':
                        is_are = 'is'
                    if flag==1:
                        word+=f"Meanwhile, it has vivid {word_3}, which {is_are} perhaps dog's {word_3}. "
                    if flag==0:
                        word += f"it has vivid {word_3}, which {is_are} perhaps dog's {word_3}. "

                if word_4 != '':
                    is_are = 'are'
                    if word_4 == 'nose':
                        is_are = 'is'
                    if flag==1:
                        word+=f"Meanwhile, it has {word_4}, which {is_are} something like dog's {word_4}. "
                    if flag==0:
                        word += f"it has {word_4}, which {is_are} something like dog's {word_4}. "

                if word_5 != '':
                    is_are = 'are'
                    if word_5 == 'nose':
                        is_are = 'is'
                    if flag==1:
                        word+=f"Meanwhile, it has {word_5}, which {is_are} perhaps dog's {word_5}. "
                    if flag==0:
                        word += f"it has {word_5}, which {is_are} perhaps dog's {word_5}. "



                for n in range(3):
                    if dog_cdf[n]>0.5 and delta[n]<=0.2:
                        word += f"Although its {word_list[n]} {is_list[n]} a little confusing. "
                    # if 0.2 < dog_cdf[n] < 0.5 and delta[n] <= 0.2:
                    #     word+=f'In addition, it seems to have {word_list[n]}, which {is_list[n]} a little confusing.'

        if label_true[0,1]<label_true[0,0]:
            delta=-delta
            if np.max(delta) < 0.2:
                word += 'It may be a cat, but I am not sure.'
            else:
                if np.max(delta) > 0.5:
                    word += 'I am sure it is a cat mainly because '
                else:
                    word += 'It is probably a cat mainly because '

                for m in range(len(word_list)):
                    if cat_cdf[m] > 0.5 and delta[m] > 0.5:
                        if word_1=='':
                            word_1+=word_list[m]
                        else:
                            word_1 +=' and '
                            word_1+=word_list[m]
                    if cat_cdf[m] > 0.5 and 0.35 < delta[m] <= 0.5:
                        if word_2 == '':
                            word_2 += word_list[m]
                        else:
                            word_2+= ' and '
                            word_2 += word_list[m]
                    if cat_cdf[m] > 0.5 and 0.2 < delta[m] <= 0.35:
                        if word_3 == '':
                            word_3 += word_list[m]
                        else:
                            word_3 += ' and '
                            word_3 += word_list[m]
                    if 0.2 < cat_cdf[m] < 0.5 and 0.35 < delta[m] <= 0.5:
                        if word_4 == '':
                            word_4 += word_list[m]
                        else:
                            word_4 += ' and '
                            word_4 += word_list[m]
                    if 0.2 < cat_cdf[m] < 0.5 and 0.2 < delta[m] <= 0.35:
                        if word_5 == '':
                            word_5 += word_list[m]
                        else:
                            word_5 += ' and '
                            word_5 += word_list[m]
                print(word_1,word_2,word_3,word_4,word_5)

                flag = 0
                if word_1 != '':
                    is_are = 'are'
                    if word_1 == 'nose':
                        is_are = 'is'
                    word += f"it has vivid {word_1}, which {is_are} cat's {word_1} obviously. "
                    flag = 1
                if word_2 != '':
                    is_are = 'are'
                    if word_2 == 'nose':
                        is_are = 'is'
                    if flag == 1:
                        word += f"Meanwhile, it has vivid {word_2}, which {is_are} something like cat's {word_2}. "
                    if flag == 0:
                        word += f"it has vivid {word_2}, which {is_are} something like cat's {word_2}. "

                if word_3 != '':
                    is_are = 'are'
                    if word_3 == 'nose':
                        is_are = 'is'
                    if flag == 1:
                        word += f"Meanwhile, it has vivid {word_3}, which {is_are} perhaps cat's {word_3}. "
                    if flag == 0:
                        word += f"it has vivid {word_3}, which {is_are} perhaps cat's {word_3}. "

                if word_4 != '':
                    is_are = 'are'
                    if word_4 == 'nose':
                        is_are = 'is'
                    if flag == 1:
                        word += f"Meanwhile, it has {word_4}, which {is_are} something like cat's {word_4}. "
                    if flag == 0:
                        word += f"it has {word_4}, which {is_are} something like cat's {word_4}. "

                if word_5 != '':
                    is_are = 'are'
                    if word_5 == 'nose':
                        is_are = 'is'
                    if flag == 1:
                        word += f"Meanwhile, it has {word_5}, which {is_are} perhaps cat's {word_5}. "
                    if flag == 0:
                        word += f"it has {word_5}, which {is_are} perhaps cat's {word_5}. "

            for n in range(3):
                    if cat_cdf[n] > 0.5 and delta[n] <= 0.2:
                        word += f"Although its {word_list[n]} {is_list[n]} a little confusing. "
                    # if 0.2 < cat_cdf[n] < 0.5 and delta[n] <= 0.2:
                    #     word+=f'In addition, it seems to have {word_list[n]}, which {is_list[n]} a little confusing. '
        print(word)
        print(dog_cdf)
        print(cat_cdf)
        result=np.zeros([6])
        result[0]=dog_cdf[1]
        result[1]=dog_cdf[0]
        result[2]=cat_cdf[0]
        result[3]=cat_cdf[1]
        result[4]=cat_cdf[2]
        result[5]=dog_cdf[2]
        plot_radian(result,file_name)

def identify_adverse(conv_out,reshape_num):
    '''
    identify adverserial expample
    :param conv_out:
    :param reshape_num:
    :return:
    '''
    config.dataset_situation = 'val'
    adverse_sample = torch.from_numpy(
        np.load(r'D:\pycharm project\VAE_PDE\fake_picture\cat-fake-dog-PGD-0.05.npy').astype(np.float32))
    true_sample = torch.from_numpy(
        np.load(r'D:\pycharm project\VAE_PDE\fake_picture\cat-true-dog-PGD-0.05.npy').astype(np.float32))
    attack_accuracy=0
    defence_accuray=0
    for i in range(100):
        print('----------第%d次迭代----------' % (i))
        image = adverse_sample[i]
        image_true=true_sample[i]
        image = Variable(image.to(device), requires_grad=True)
        label_true, x_feature = use_model(image)
        _, prediction = torch.max(label_true.data, 1)
        if prediction==1:
            attack_accuracy+=1
        print(transform_pro(label_true))
        act = conv_out.features
        x_feature = act.to(device).reshape(reshape_num)
        feature_space = x_feature
        pic = image.cpu().data
        img = transform_raw_picture(pic)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # if i==6:
        #         pic=image.cpu().data
        #         img=transform_raw_picture(pic)
        #         plt.imshow(img)
        #         break

        cdf_record = []
        for i in ['eye', 'nose', 'leg']:
            for j in ['dog', 'cat']:
                config.position_space = i
                config.position_animal = j
                _, _, space_index, space_value = get_position(conv_out_2_0, 5, show_picture=False)
                space_index = np.array(space_index, dtype=int)
                #This file can be generated by running the function val_distribution with modifying dataset_situation and save_name
                space = np.load(f'result_save/{config.position_animal}_space_2000.npy')

                eye_space = space[:, space_index]
                sum = 0
                for k in range(eye_space.shape[1]):
                    sum += eye_space[:, k] * space_value[k]
                mean = sum / eye_space.shape[1]

                eye_space = feature_space[space_index].cpu().data.numpy()
                sum = 0
                for k in range(eye_space.shape[0]):
                    sum += eye_space[k] * space_value[k]
                mean_pic = sum / eye_space.shape[0]
                cdf_record.append(get_cdf_ratio(mean_pic, mean))
                print(f'{j},{i},{get_cdf_ratio(mean_pic, mean)}')
        dog_cdf=np.array(cdf_record)[[0,2,4]]
        if (dog_cdf>0.99).any() and np.sum(dog_cdf>0.9)>=2:
            defence_accuray+=1

        print(attack_accuracy,defence_accuray)

def get_inverse_position(conv_out,show_picture=False):
    '''
    visualize the semantic space
    :param conv_out:
    :param show_picture:
    :return:
    '''
    position = torch.from_numpy(np.load(f'result_save/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))
    position_origin=torch.from_numpy(np.load(f'result_save/{config.position_space}_save_{config.position_animal}_origin.npy').astype(np.float32))
    if show_picture==True:
        for i in range(position.shape[0]):
            plt.figure(1)
            image_plot = transform_raw_picture(position[i].cpu().data)
            plt.imshow(image_plot)


    feature_save=[]
    feature_save_origin=[]
    for i in range(position.shape[0]):
        image = position[i]
        image_origin=position_origin[i]


        label_true, x_feature = use_model(image.cuda())
        act = conv_out.features
        x_feature = act.to(device).reshape(512 )

        feature_save.append(act.data.numpy())

        label_true, x_feature = use_model(image_origin.cuda())
        act = conv_out.features
        feature_save_origin.append(act.data.numpy())
        x_feature_origin = act.to(device).reshape(512)

        if show_picture == True:
            plt.figure(2)
            plt.subplot(10,10,i+1)
            plt.plot((x_feature).cpu().data.numpy())
            plt.plot((-x_feature_origin).cpu().data.numpy())


    feature_matrix=np.array(feature_save)
    feature_matrix_origin = np.array(feature_save_origin)
    feature_PCA=position_PCA(feature_matrix,n_components=10)
    feature_PCA_origin=position_PCA(feature_matrix_origin,n_components=10)

    #cat space visualization
    eye_feature=np.zeros([1,512,1,1])
    error=feature_PCA_origin-feature_PCA
    sort_index_cat=[145,263,391,461,463]
    for i in range(5):
        eye_feature[0,sort_index_cat[i],0,0]=error[0,sort_index_cat[i],0,0]
    eye_feature=eye_feature*30/np.max(eye_feature)
    plt.style.use('ggplot')
    plt.figure(4, figsize=(4, 4), dpi=300)
    plt.bar(range(512), eye_feature[0,:,0,0])

    for i in range(len(sort_index_cat)):
        plt.text(sort_index_cat[i],
                 (feature_PCA_origin[0, :, 0, 0].cpu().data.numpy() - feature_PCA[0, :, 0, 0].cpu().data.numpy())[
                     sort_index_cat[i]] + 0.25, '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
                 fontproperties='Arial', fontsize=8)

    plt.xlabel('Neuron Index', font1)
    plt.ylabel('Scores of 1st PC', font1)
    plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}_only.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}_only.pdf',
                bbox_inches='tight', dpi=300)

    plt.show()
    inverse_pic, inverse_feature = picture_inverse(use_model, torch.from_numpy(eye_feature.astype((np.float32))).to(device), max_epoch=4000)
    inverse_pic_plot = transform_raw_picture(inverse_pic)



    plt.style.use('default')
    plt.figure(5, dpi=300)
    plt.axis('off')
    plt.imshow(np.clip(inverse_pic_plot, 0, 1))
    plt.savefig(f'PPT_fig/inverse_pic_{config.position_space}_save_{config.position_animal}_only.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/inverse_pic_{config.position_space}_save_{config.position_animal}_only.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

    inverse_pic, inverse_feature = picture_inverse(use_model, feature_PCA.to(device),max_epoch=4000)
    inverse_pic_plot = transform_raw_picture(inverse_pic)

    plt.style.use('default')
    plt.figure(5, dpi=300)
    plt.axis('off')
    plt.imshow(np.clip(inverse_pic_plot, 0, 1))
    plt.savefig(f'PPT_fig/inverse_pic_{config.position_space}_save_{config.position_animal}_no.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/inverse_pic_{config.position_space}_save_{config.position_animal}_no.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

    inverse_pic, inverse_feature = picture_inverse(use_model, feature_PCA_origin.to(device), max_epoch=4000)
    inverse_pic_plot = transform_raw_picture(inverse_pic)

    plt.style.use('default')
    plt.figure(5, dpi=300)
    plt.axis('off')
    plt.imshow(np.clip(inverse_pic_plot, 0, 1))
    plt.savefig(f'PPT_fig/inverse_pic_{config.position_space}_save_{config.position_animal}_yes.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/inverse_pic_{config.position_space}_save_{config.position_animal}_yes.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

def plot_N():
    '''
    plot the results of different numbers of Ns
    :return:
    '''
    data_num = [2, 5, 20, 50, 100, 300, 500]
    PCA_525=np.load('PPT_fig/paper/PCA_save_cat_500_525.npy')

    PCA_324=np.load('PPT_fig/paper/PCA_save_cat_500_324.npy')

    PCA_1101=np.load('PPT_fig/paper/PCA_save_cat_500_1101.npy')
    error=[]
    for i in range(7):
        average=(PCA_525[i]+PCA_324[i]+PCA_1101[i])/3
        relative_error=((np.abs(PCA_525[i]-average)+np.abs(PCA_324[i]-average)+np.abs(PCA_1101[i]-average)).sum())/(3*average.sum())
        error.append(relative_error)
        print(relative_error)
    fig = plt.figure(1, figsize=(3, 1.5), dpi=300)
    x = np.linspace(0, 1, 7)
    plt.plot(x,error,c='blue',zorder=1)
    plt.scatter(x, error,marker='x', c='red',s=15,zorder=2)
    plt.xticks(x, data_num, fontproperties='Arial', size=7)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8], ['0%', '20%', "40%", "60%", "80%"], fontproperties='Arial', size=7)
    # for i in range(len(data_num)):
    #     plt.text(data_num[i], error[i] + 0.02, '%s' % round(error[i], 3), ha='center',
    #              fontproperties='Arial', fontsize=8)
    # plt.savefig('PPT_fig/paper/PCA_number_save.tiff', bbox_inches='tight',  dpi=300)
    # plt.savefig('PPT_fig/paper/PCA_number_save.pdf', bbox_inches='tight', dpi=300)
    plt.show()


#--------------------main function----------------



#plot_N()

config.position_target='eye'
config.position_space='eye'
config.position_animal='cat'
use_model = feature_net('vgg', dim=512, n_classes=2)  # for vgg dim=512
use_model.load_state_dict(torch.load("CNN_model_save/VGG_224.pth"))
use_model.eval()
use_model.to(device)
conv_out_2_0 = LayerActivations(use_model.feature[2], 0)


def plot_lime_PCA():
    font1 = {'family': 'Arial',
             'weight': 'normal',
             # "style": 'italic',
             'size': 16,
             }

    config.PCA_animal='cat'
    config.PCA_data_num=300
    config.PCA_main=1
    generate_feature_matrix(525,lime=False)
    PCA_feature=load_PCA()
    plt.figure(1, figsize=(4, 4), dpi=300)
    plt.bar(range(512), PCA_feature.cpu().data.numpy().reshape(512), color='blue')
    plt.xlabel('Neuron Index', font1)
    plt.ylabel('Activation after PCA', font1)
    # plt.savefig(f'PPT_fig/neuron_{config.PCA_animal}_{config.PCA_data_num}_lime_{config.PCA_main}.tiff',
    #             bbox_inches='tight', dpi=300)
    # plt.savefig(f'PPT_fig/neuron_{config.PCA_animal}_{config.PCA_data_num}_lime_{config.PCA_main}.pdf',
    #             bbox_inches='tight', dpi=300)
    plt.show()


    config.TV_coef=2
    plt.style.use('default')
    pic_new,feature_prediction=picture_inverse(use_model,PCA_feature,max_epoch=4000)
    img = transform_raw_picture(pic_new)
    plt.figure(1,dpi=300)
    plt.imshow(np.clip(img,0,1))
    plt.axis('off')
    plt.savefig(f'PPT_fig/inverse_{config.PCA_animal}_{config.PCA_data_num}_no_lime_{config.PCA_main}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/inverse_{config.PCA_animal}_{config.PCA_data_num}_no_lime_{config.PCA_main}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

#get_inverse_position(conv_out_2_0)
#cut_position(use_model)
#sort_index_origin,sort_index_position,space_index,space_value=get_position(conv_out_2_0,5,show_picture=True)
#plot_distribution(space_index,space_value,picture='small')
#identify_adverse(conv_out_2_0,512)
print_word(conv_out_2_0,512)
#val_distribution(conv_out_2_0,512)
#cat_space,cat_space_pic=get_space(conv_out_2_0,512,sort_index_origin,sort_index_position,add_position='position')


