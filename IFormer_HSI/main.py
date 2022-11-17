import numpy as np
import scipy.io as sio
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
import auxil
from IF_Transformer import IF_Transformer
from Utils import load_data


def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 75)) * (0.1 ** (epoch // 125))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(trainloader, model, criterion, optimizer, epoch, use_cuda=True):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[batch_idx] = loss.item()
        accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()
    return (np.average(losses), np.average(accs))


def data_test(testloader, model, criterion, epoch, use_cuda=True):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            losses[batch_idx] = criterion(outputs, targets).item()
            accs[batch_idx] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()

    return (np.average(losses), np.average(accs))


def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda: inputs = inputs.cuda()
            # [predicted.append(a) for a in model(inputs).data.cpu().numpy()]
            [predicted.append(F.softmax(a, dim=0).cpu().numpy()) for a in model(inputs).data]

    return np.array(predicted)


def main(spatial_size):
    hsi = sio.loadmat('./datasets/Indian_pines_corrected.mat')
    hsi = hsi['indian_pines_corrected']
    h, w, c = hsi.shape
    print('Hyperspectral data shape: ', hsi.shape)

    gt = sio.loadmat('./datasets/Indian_pines_gt.mat')
    gt = gt['indian_pines_gt']
    print('Label shape: ', gt.shape)

    hsi = hsi.reshape(h * w, c)
    hsi = StandardScaler().fit_transform(hsi)
    hsi = hsi.reshape(h, w, c)
    num_class = len(np.unique(gt)) - 1

    train_loader, test_loader, val_loader, num_classes, n_bands = load_data(hsi,
                                                                            gt,
                                                                            spatialsize=spatial_size,
                                                                            tr_percent=0.10,
                                                                            numclass=num_class,
                                                                            val_percent=0.01)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IF_Transformer(image_size=spatial_size, in_channels=c, num_classes=num_class).to(device)
    model = model.cuda()
    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    best_acc = -1
    for epoch in range(150):
        adjust_learning_rate(optimizer, epoch, lr)

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        # scheduler.step()
        test_loss, test_acc = data_test(val_loader, model, criterion, epoch)

        print("EPOCH", epoch, "TRAIN LOSS", train_loss, "TRAIN ACCURACY", train_acc, end=',')
        print("LOSS", test_loss, "ACCURACY", test_acc)
        # save model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, "best_model.pth.tar")

    checkpoint = torch.load("best_model.pth.tar")
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # test_loss, test_acc = test(test_loader, model, criterion, epoch)
    print("FINAL:      ACCURACY", best_acc)

    prediction = predict(test_loader, model, criterion, True)
    prediction = np.argmax(prediction, axis=1)
    classification, confusion, results = auxil.reports(prediction, np.array(test_loader.dataset.__labels__()))

    # return np.array(results[0:3])
    return results


if __name__ == '__main__':
    num = 1
    OA = []
    AA = []
    KAPPA = []
    classes = 16
    CA = np.zeros([classes, num])
    spatial_size = 9

    for i in range(num):
        print('iteration {}'.format(i + 1))
        results = main(spatial_size)
        if i == 0:
            OA = results[0]
            AA = results[1]
            KAPPA = results[2]
        else:
            OA = np.hstack([OA, results[0]])
            AA = np.hstack([AA, results[1]])
            KAPPA = np.hstack([KAPPA, results[2]])

        CA[:, i] = results[3:len(results)]
        print('oa is :', results[0], ' aa is :', results[1], ' kappa is :', results[2])
        print('ca are :')
        print(results[3:len(results)])
        print('----------------------------------------------------------------------')

    OA_average = np.mean(OA)
    OA_std = np.std(OA)
    AA_average = np.mean(AA)
    AA_std = np.std(AA)
    KAPPA_average = np.mean(KAPPA)
    KAPPA_std = np.std(KAPPA)
    CA_average = np.mean(CA, axis=1)
    CA_std = np.std(CA, axis=1)

    average_values = list(np.round(np.array([OA_average, AA_average, KAPPA_average] + list(CA_average)), 2))
    std_values = list(np.round(np.array([OA_std, AA_std, KAPPA_std] + list(CA_std)), 2))
    print('each average value : ', average_values)
    print('each std value : ', std_values)