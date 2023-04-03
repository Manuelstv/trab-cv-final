import argparse
from spherenet import OmniMNIST, OmniFashionMNIST, OmniCustom
from spherenet import SphereConv2D, SphereMaxPool2D
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

#https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    ##assert bb1['x1'] < bb1['x2']
    #assert bb1['y1'] < bb1['y2']
    #assert bb2['x1'] < bb2['x2']
    #assert bb2['y1'] < bb2['y2']
    if (bb2['x1'] > bb2['x2'] or bb2['y1'] > bb2['y2']):
        #print(bb2)
        return 0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou.item()


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
        #self.conv7 = SphereConv2D(1024, 2048, stride=1)
        #self.pool7 = SphereMaxPool2D(stride=2)
        #self.conv8 = SphereConv2D(2048, 2048, stride=1)
        #self.pool8 = SphereMaxPool2D(stride=2)

        self.classifier = nn.Sequential(nn.Linear(8192, 4096), nn.Dropout(0.5), nn.Linear(4096, 4))
        self.bb = nn.Sequential(nn.Linear(8192, 4096), nn.Dropout(0.5), nn.Linear(4096, 4))

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = F.relu(self.pool5(self.conv5(x)))
        x = F.relu(self.pool6(self.conv6(x)))
        #x = F.relu(self.pool7(self.conv7(x)))
        #x = F.relu(self.pool8(self.conv8(x)))
        #print(x.size())
        x = x.view(-1, 8192)  # flatten, [B, C, H, W) -> (B, C*H*W)
        
        x_box = self.bb(x)
        x_class = self.classifier(x)
        
        return x_class, x_box 
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.fc = nn.Linear(8192, 4096)
        self.classifier = nn.Sequential(nn.Linear(8192, 4096), nn.Dropout(0.5), nn.Linear(4096, 4))
        self.bb = nn.Sequential(nn.Linear(8192, 4096), nn.Dropout(0.5), nn.Linear(4096, 4))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv6(x), 2))
        #print(x.size())
        x = x.view(-1, 8192) 
        
        x_box = self.bb(x)
        x_class = self.classifier(x)
        
        return x_class, x_box 

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    total_box_loss = 0
    mean_iou = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if data.dim() == 3:
            data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)

        y_pred, z_pred = model(data.to(device))
        z = torch.stack((target["x_min"], target["y_min"], target["x_max"], target["y_max"]), 1).to(device)
        class_loss = F.cross_entropy(y_pred, target["labels"].long().to(device))  

        box_loss = F.mse_loss(z_pred, z)

        #for b in range(z.shape[0]):
        #    mean_iou += get_iou({'x1': z[0][0], 'x2': z[0][2], 'y1': z[0][1], 'y2': z[0][3]}, {'x1': z_pred[0][0], 'x2': z_pred[0][2], 'y1': z_pred[0][1], 'y2': z_pred[0][3]})
        total_box_loss += box_loss
        
        total_loss = box_loss*0.9 + class_loss*0.1
        total_loss.backward()
        
        optimizer.step()
        train_loss += total_loss.item()
        
        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t box loss {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), total_loss.item(), box_loss.item()))
     
    train_loss /= len(train_loader.dataset)
    total_box_loss /=  len(train_loader.dataset)
    mean_iou /= len(train_loader.dataset)
    
    print('\n Total box Loss: ', total_box_loss.item())
    print('\n Train Loss: ', train_loss)
    #print('\n Mean IOU: ', mean_iou)

    return train_loss


def test(args, model, device, test_loader):
    model.eval()
    total_loss = 0
    correct_0, correct_3, correct_6 = 0, 0, 0
    test_loss = 0
    total_box_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to(device), target.long().to(device)
            if data.dim() == 3:
                data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)

            y_pred, z_pred = model(data.to(device))
            y_pred = y_pred.squeeze(1)
            z = torch.stack((target["x_min"], target["y_min"], target["x_max"], target["y_max"]), 1).to(device)

            class_loss = F.cross_entropy(y_pred, target["labels"].long().to(device))

            box_loss = F.mse_loss(z_pred, z)
            
            total_box_loss += box_loss
        
            total_loss = box_loss*0.9 + class_loss*0.1
            test_loss += total_loss.item()

            pred = y_pred.max(1, keepdim=True)[1] # get the index of the max log-probability

            for b in range(z.shape[0]):
                if(get_iou({'x1': z[0][0], 'x2': z[0][2], 'y1': z[0][1], 'y2': z[0][3]}, {'x1': z_pred[0][0], 'x2': z_pred[0][2], 'y1': z_pred[0][1], 'y2': z_pred[0][3]})>0 and pred[b]==target["labels"][b]):
                    correct_0 += 1
                if(get_iou({'x1': z[0][0], 'x2': z[0][2], 'y1': z[0][1], 'y2': z[0][3]}, {'x1': z_pred[0][0], 'x2': z_pred[0][2], 'y1': z_pred[0][1], 'y2': z_pred[0][3]})>0.3 and pred[b]==target["labels"][b]):
                    correct_3 += 1
                if(get_iou({'x1': z[0][0], 'x2': z[0][2], 'y1': z[0][1], 'y2': z[0][3]}, {'x1': z_pred[0][0], 'x2': z_pred[0][2], 'y1': z_pred[0][1], 'y2': z_pred[0][3]})>0.6 and pred[b]==target["labels"][b]):
                    correct_6 += 1

            #correct += pred.eq(target["labels"].long().to(device).view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    total_box_loss /= len(test_loader.dataset)
    print(correct_0, correct_3, correct_6, len(test_loader.dataset))
    #print('\nTest set: Average loss: {:.4f}, box loss {:.4f} Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, box_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))
    
    return(test_loss)

def main():
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='OmniCustom',
                        help='dataset for training, options={"FashionMNIST", "MNIST", "OmniCustom"}')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
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

    for k in range(0, 3):
        np.random.seed(args.seed)
        if args.data == 'FashionMNIST':
            train_dataset = OmniFashionMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, img_std=255, train=True)
            test_dataset = OmniFashionMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, img_std=255, train=False, fix_aug=True)
        if args.data == 'OmniCustom':
            train_dataset = OmniCustom(root=f'/home/msnuel/trab-final-cv/cross_val/dataset_fold_{k}/train', fov=160, flip=True, h_rotate=True, v_rotate=True, img_std=255, train=True)
            val_dataset = OmniCustom(root=f'/home/msnuel/trab-final-cv/cross_val/dataset_fold_{k}/val', fov=160, flip=True, h_rotate=True, v_rotate=True, img_std=255, train=False, fix_aug=True)
            test_dataset = OmniCustom(root=f'/home/msnuel/trab-final-cv/cross_val/dataset_fold_{k}/test', fov=160, flip=True, h_rotate=True, v_rotate=True, img_std=255, train=False, fix_aug=True)
        elif args.data == 'MNIST':
            train_dataset = OmniMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=True)
            test_dataset = OmniMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=False, fix_aug=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
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
    
        loss_train=[]
        loss_train_cnn=[]
        loss_test=[]
        loss_test_cnn=[]

        for epoch in range(1, args.epochs + 1):
            ## SphereCNN
            print('{} Sphere CNN {}'.format('='*10, '='*10))
            loss_train.append(train(args, sphere_model, device, train_loader, sphere_optimizer, epoch))
            loss_test.append(test(args, sphere_model, device, val_loader))

            # Conventional CNN
            print('{} Conventional CNN {}'.format('='*10, '='*10))
            loss_train_cnn.append(train(args, model, device, train_loader, optimizer, epoch))
            loss_test_cnn.append(test(args, model, device, val_loader))
            #if epoch % args.save_interval == 0:
            #    torch.save(model.state_dict(), 'model.pkl')
    
        test(args, sphere_model, device, test_loader)
        test(args, model, device, test_loader)
        print(loss_train)
        print(loss_test)
        print(loss_train_cnn)
        print(loss_test_cnn)


if __name__ == '__main__':
    main()
