import torch
import numpy as np
import random
import auxil
from hyper_pytorch import HyperData
import torch.nn.functional as F



def load_data(data, labels, spatialsize, numclass, tr_percent, val_percent, tr_bsize=20, te_bsize=300, use_val=True):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    h, w = labels.shape[0], labels.shape[1]
    labels = labels.reshape(h * w)
    bands = data.shape[-1]

    train_num = [15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50]
    num_class = np.max(labels)

    train_label = np.zeros_like(labels)
    test_label = np.zeros_like(labels)
    for i in range(num_class):
        r = random.random
        random.seed(13)
        index = np.where(labels == i + 1)[0]
        random.shuffle(index, r)
        train_index = index[:train_num[i]]
        test_index = index[train_num[i]:]
        train_label[train_index] = labels[train_index]
        test_label[test_index] = labels[test_index]

    labels = labels.reshape(h, w)
    train_label = train_label.reshape(h, w)
    test_label = test_label.reshape(h, w)

    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = auxil.chooose_train_and_test_point(
        train_label, test_label, labels, num_class)
    margin = int((spatialsize - 1) / 2)
    zeroPaddedX = auxil.padWithZeros(data, margin=margin)
    print('\n... ... create train & test data ... ...')
    x_train, x_test = auxil.train_and_test_data(zeroPaddedX,
                                                band=data.shape[-1],
                                                train_point=total_pos_train,
                                                test_point=total_pos_test,
                                                true_point=total_pos_true,
                                                patch=spatialsize)
    y_train, y_test = auxil.train_and_test_label(number_train, number_test, number_true, num_class)
    del total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true, zeroPaddedX
    if use_val:
        x_val, x_test, y_val, y_test = auxil.split_data(x_test, y_test, val_percent)
    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"), y_train))
    test_hyper = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"), y_test))
    if use_val:
        val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"), y_val))
    else:
        val_hyper = None
    # kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=tr_bsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=te_bsize, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=te_bsize, shuffle=False, **kwargs)

    return train_loader, test_loader, val_loader, num_class, bands
