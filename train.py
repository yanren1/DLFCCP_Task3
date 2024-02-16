import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloader import SampleDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, models
from torchvision.transforms import transforms

from tqdm import tqdm
import numpy as np
from backbone.model import MyResnet18,simpleMLP,Myregnet16,SparseAutoencoder,CNNAutoencoder
from backbone.ghostnetv2_torch import MyGhostnetv2,ghostnetv2
from backbone.capsnet import CapsNet,CapsConfig
from PIL import Image, ImageDraw, ImageFont
import time
from torch.autograd import Variable
from tqdm import tqdm



def save_model(model_save_pth,model, epoch,train_ce,accuracy,model_type=''):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename = 'model_{}_epoch_{}_train_ce_{:0.2e}_val_Acc_{:0.2e}.pt'.format(model_type,
                                                                                     epoch,
                                                                                     train_ce,
                                                                                     accuracy,)

    filename = os.path.join(model_save_pth,filename)
    torch.save(model.state_dict(), filename)


def train():

    debug = False
    use_pretrain = False


    # split train and val set
    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop(size = 28),
        # transforms.ColorJitter(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    # train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    # val_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_val)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    batch_size = int(512)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    model_type = ['ae','sparse_ae','cnn_ae','ghostnet','resnet','regnet','CapsNet']
    model_type = model_type[-1]

    # backbone
    if model_type =='resnet':
        backbone = MyResnet18(pretrained=True,num_classes = 10,in_channels = 3).cuda()
    elif model_type == 'regnet':
        backbone = Myregnet16(pretrained=True,num_classes = 10,in_channels = 3).cuda()
    elif model_type == 'ghostnet':
        # backbone = ghostnetv2(num_classes=1000, width=1.0, dropout=0.1,args=None).cuda()
        backbone = MyGhostnetv2(num_classes=10, in_channels=3,width=1, dropout=0.1).cuda()
    elif model_type == 'ae':
        backbone = simpleMLP(in_channels=784,
                             # hidden_channels=[128,784,18],
                             hidden_channels=[784,784,784,784,784,18],
                             norm_layer=nn.BatchNorm1d,
                             dropout=0, inplace=False, use_sigmoid=False).cuda()
    elif model_type == 'sparse_ae':
        backbone = SparseAutoencoder(in_channels=784,
                             # hidden_channels=[128,784,18],
                             encoder_channels=[1024,2048],
                             decoder_channels=[1024,784,18],

                             norm_layer=nn.BatchNorm1d,
                             dropout=0, inplace=False).cuda()
    elif model_type == 'cnn_ae':
        backbone = CNNAutoencoder(in_channels=1,
                       encoder_channels=[128, 64, 32, 16],
                       decoder_channels=[16, 32, 64, 128],
                       num_features=1024,
                       num_classes=10,
                       norm_layer=nn.BatchNorm2d,
                       dropout=0.1).cuda()

    elif model_type == 'CapsNet':
        model_config = CapsConfig('cifar10')
        # model_config = CapsConfig('mnist')
        backbone = CapsNet(model_config).cuda()


    criterion = nn.CrossEntropyLoss().cuda()
    criterion_val = nn.CrossEntropyLoss().cuda().eval()

    # try read pre-train model
    if use_pretrain:
        weights_pth = 'final.pt'
        try:
            backbone.load_state_dict(torch.load(weights_pth))
        except:
            print(f'No {weights_pth}')

    # set lr,#epoch, optimizer and scheduler
    lr = 1e-3
    num_epoch = 50
    optimizer = optim.Adam(
        backbone.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=8e-5)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_save_pth = os.path.join('model_saved', model_type + '_'+current_time)
    os.mkdir(model_save_pth)
    writer = SummaryWriter(model_save_pth)

    # start training
    backbone.train()
    for epoch in range(num_epoch):
        loss_list = []
        for sample, target in tqdm(train_loader):
            backbone.zero_grad()
            # print(sample.shape, target.shape)
            # sample, target = sample.cuda(), target.cuda()
            # print(sample, target)

            if model_type =='sparse_ae':
                sample, target = sample.cuda(), target.cuda()
                output,activations = backbone(sample)
                loss = criterion(output, target) + backbone.compute_sparsity_loss(activations) * 0.1

            elif model_type =='CapsNet':
                target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
                sample, target = Variable(sample), Variable(target)
                sample, target = sample.cuda(), target.cuda()

                output, reconstructions, masked = backbone(sample)
                loss = backbone.loss(sample, output, target, reconstructions).cuda()

            else:
                sample, target = sample.cuda(), target.cuda()
                output = backbone(sample)
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        scheduler.step()

        if (epoch+1) % 1 == 0 or epoch==0:
            # print(f'\r Epoch:{epoch} ce loss = {np.mean(loss_list)} ,lr = {optimizer.param_groups[0]["lr"]}     ', end = ' ')
            tqdm.write(f'\r Epoch:{epoch} ce loss = {np.mean(loss_list)} ,lr = {optimizer.param_groups[0]["lr"]}     ',)
            writer.add_scalar('Training CE Loss', np.mean(loss_list), epoch)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], epoch)

        # valing and save
        if (epoch+1) % 10 == 0 or epoch==0:
            # print('Valing.....')
            tqdm.write('Valing.....')
            val_loss_list = []
            backbone.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for val_sample, val_target in val_loader:
                    # val_sample, val_target = val_sample.cuda(), val_target.cuda()

                    if model_type == 'sparse_ae':
                        val_sample, val_target = val_sample.cuda(), val_target.cuda()
                        output,activations = backbone(val_sample)
                        val_loss = criterion_val(output, val_target)+ backbone.compute_sparsity_loss(activations) * 0.1

                        val_loss_list.append(val_loss.item())
                        _, predicted = torch.max(output.data, 1)
                        total += val_target.size(0)
                        correct += (predicted == val_target).sum().item()

                    elif model_type == 'CapsNet':
                        val_target = torch.sparse.torch.eye(10).index_select(dim=0, index=val_target)
                        val_sample, val_target = Variable(val_sample), Variable(val_target)
                        val_sample, val_target = val_sample.cuda(), val_target.cuda()

                        output, reconstructions, masked = backbone(val_sample)
                        val_loss = backbone.loss(val_sample, output, val_target, reconstructions)

                        val_loss_list.append(val_loss.item())
                        total += val_sample.shape[0]
                        correct += sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                                       np.argmax(val_target.data.cpu().numpy(), 1))

                    else:
                        val_sample, val_target = val_sample.cuda(), val_target.cuda()
                        output = backbone(val_sample)
                        val_loss = criterion_val(output, val_target)

                        val_loss_list.append(val_loss.item())
                        _, predicted = torch.max(output.data, 1)
                        total += val_target.size(0)
                        correct += (predicted == val_target).sum().item()


            Train_ce = np.mean(loss_list)
            val_ce = np.mean(val_loss_list)
            accuracy = correct / total

            writer.add_scalar('Validation ce', val_ce, epoch + 1)
            writer.add_scalar('Validation accuracy', accuracy, epoch + 1)

            print(f'VAL Epoch:{epoch} Train ce = {Train_ce}, '
                  f'val ce = {val_ce} , val accuracy = {accuracy}')
            tqdm.write(f'VAL Epoch:{epoch} Train ce = {Train_ce}, '
                  f'val ce = {val_ce} , val accuracy = {accuracy}')

            print()
            # save_model(model_save_pth,backbone, epoch, Train_ce, accuracy)
            backbone.train()

    torch.save(backbone.state_dict(), os.path.join(model_save_pth,'final.pt'))
    # dummy_input = torch.randn([1, 1, 28, 28], requires_grad=True).cuda()
    # torch.onnx.export(backbone,  # model being run
    #                   dummy_input,  # model input (or a tuple for multiple inputs)
    #                   os.path.join(model_save_pth,'final.onnx'),  # where to save the model
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=10,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['modelInput'],  # the model's input names
    #                   output_names=['modelOutput'],  # the model's output names
    #                   dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
    #                                 'modelOutput': {0: 'batch_size'}})
    writer.flush()
    writer.close()

if __name__ == '__main__':
    train()