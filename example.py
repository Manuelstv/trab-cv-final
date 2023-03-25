import argparse
from spherenet import OmniMNIST, OmniFashionMNIST, OmniCustom
from spherenet import SphereConv2D, SphereMaxPool2D
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SphereNet(nn.Module):
    def __init__(self):
        super(SphereNet, self).__init__()
        #256 x 256 x 32
        self.conv1 = SphereConv2D(1, 32, stride=1)
        self.pool1 = SphereMaxPool2D(stride=2)
        #128 x 128 x 64
        self.conv2 = SphereConv2D(32, 64, stride=1)
        self.pool2 = SphereMaxPool2D(stride=2)
        #64 x 64 x 128
        self.conv3 = SphereConv2D(64, 128, stride=1)
        self.pool3 = SphereMaxPool2D(stride=2)
        #32 x 32 x 256
        self.conv4 = SphereConv2D(128, 256, stride=1)
        self.pool4 = SphereMaxPool2D(stride=2)
        #16 x 16 x 512
        self.conv5 = SphereConv2D(256, 512, stride=1)
        self.pool5 = SphereMaxPool2D(stride=2)
        #8 x 8 x 512 
        self.conv6 = SphereConv2D(512, 512, stride=1)
        self.pool6 = SphereMaxPool2D(stride=2)

        self.classifier = nn.Sequential(nn.BatchNorm1d(8192), nn.Linear(8192, 4096), nn.Dropout(0.25), nn.Linear(4096, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(8192), nn.Linear(8192, 4096), nn.Dropout(0.25), nn.Linear(4096, 4))

        #self.dropout = nn.Dropout(0.25)
        
        #self.fc_class1 = nn.Linear(8192, 4096)
        #self.fc_class2 = nn.Linear(4096, 4)
        
        #self.fc_box = nn.Linear(8192, 4)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = F.relu(self.pool5(self.conv5(x)))
        x = F.relu(self.pool6(self.conv6(x)))
        #print(x.size())
        x = x.view(-1, 8192)  # flatten, [B, C, H, W) -> (B, C*H*W)
        
        x_box =  self.bb(x)
        #x = torch.nn.Softmax()(self.fc(x))
        x_class = self.classifier(x)
        
        return x_class, x_box 
        #return x_class 
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc = nn.Linear(18432, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))                
        #print(x.size())
        x = x.view(-1, 18432)  # flatten, [B, C, H, W) -> (B, C*H*W)
        x = self.fc(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    total_box_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if data.dim() == 3:
            data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)

        y_pred, z_pred = model(data.to(device))
        y_pred = y_pred.squeeze(1)
        z = torch.stack((target["x"],target["y"],target["w"],target["h"]), 1).to(device)

        class_loss = F.cross_entropy(y_pred, target["labels"].long().to(device))


        box_loss = F.l1_loss(z_pred, z, reduction="none").sum(1)
        box_loss = box_loss.sum()
        
        #box_loss = F.mse_loss(z_pred, z)
        total_box_loss += box_loss
        
        total_loss = box_loss/1000 + class_loss
        #total_loss = class_loss
        class_loss.backward()
        
        optimizer.step()
        train_loss += total_loss.item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t box loss {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item(), box_loss.item()))
     
    train_loss /= len(train_loader.dataset)
    total_box_loss /=  len(train_loader.dataset)
    
    print('\n Total box Loss: ', total_box_loss)
    print('\n Train Loss: ', train_loss)


def test(args, model, device, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    test_loss = 0
    total_box_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to(device), target.long().to(device)
            if data.dim() == 3:
                data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)

            y_pred, z_pred = model(data.to(device))
            y_pred = y_pred.squeeze(1)
            z = torch.stack((target["x"],target["y"],target["w"],target["h"]), 1).to(device)

            class_loss = F.cross_entropy(y_pred, target["labels"].long().to(device))

            print(z_pred, z)
            #box_loss = F.mse_loss(z_pred, z)
            box_loss = F.l1_loss(z_pred, z, reduction="none").sum(1)
            box_loss = box_loss.sum()
            
            total_box_loss += box_loss
        
            total_loss = box_loss/1000 + class_loss
            test_loss += total_loss.item()

            #test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = y_pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target["labels"].long().to(device).view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    total_box_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, box loss {:.4f} Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, box_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='OmniCustom',
                        help='dataset for training, options={"FashionMNIST", "MNIST", "OmniCustom"}')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer, options={"adam, sgd"}')
    parser.add_argument('--lr', type=float, default=1E-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before saving model weights')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    np.random.seed(args.seed)
    if args.data == 'FashionMNIST':
        train_dataset = OmniFashionMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, img_std=255, train=True)
        test_dataset = OmniFashionMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, img_std=255, train=False, fix_aug=True)
    if args.data == 'OmniCustom':
        train_dataset = OmniCustom(root='/home/msnuel/trab-final-cv/animals/train', fov=120, flip=True, h_rotate=True, v_rotate=True, img_std=255)
        test_dataset = OmniCustom(root='/home/msnuel/trab-final-cv/animals/test', fov=120, flip=True, h_rotate=True, v_rotate=True, img_std=255, fix_aug=True)
    elif args.data == 'MNIST':
        train_dataset = OmniMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=True)
        test_dataset = OmniMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=False, fix_aug=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Train
    sphere_model = SphereNet().to(device)
    model = Net().to(device)
    if args.optimizer == 'adam':
        sphere_optimizer = torch.optim.Adam(sphere_model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        sphere_optimizer = torch.optim.SGD(sphere_model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        ## SphereCNN
        print('{} Sphere CNN {}'.format('='*10, '='*10))
        train(args, sphere_model, device, train_loader, sphere_optimizer, epoch)
        test(args, sphere_model, device, test_loader)
        if epoch % args.save_interval == 0:
            torch.save(sphere_model.state_dict(), 'sphere_model.pkl')

        # Conventional CNN
        #print('{} Conventional CNN {}'.format('='*10, '='*10))
        #train(args, model, device, train_loader, optimizer, epoch)
        #test(args, model, device, test_loader)
        #if epoch % args.save_interval == 0:
        #    torch.save(model.state_dict(), 'model.pkl')


if __name__ == '__main__':
    main()
