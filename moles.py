import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import csv
import sys
import os
import argparse
import math

def moleRecruitment(matrix, N):
    moles = []
    print(len(matrix))
    for i in range(len(matrix)):
        partialpoison = []
        for j in range(matrix.shape[-1] - 2):
            part = np.flip(matrix[i][matrix[i][:, j].argsort()], axis=0)
            partialpoison.append(part[:N])
        moles.append(partialpoison)
    moles_np = np.array(moles)
    return moles_np

def selectCombo(moles, past_classes, curr_classes, all_classes):
    percentiles = []
    for i in past_classes:  # attacked
        for j in curr_classes:  # confounding
            if i != j: #don't need this since curr != past
                spread = moles[all_classes.index(j)][all_classes.index(i)][:, all_classes.index(i)]
                percentiles.append([np.percentile(spread, 99), i, j])
    percentiles = sorted(percentiles, key=lambda x: float(x[0]), reverse=True)
    print(percentiles[0])
    return percentiles[0][1], percentiles[0][2], round(percentiles[0][0], 2)

def moleSet(N, id, n_experiences, attacked, confounding, moles, table, curr_classes, all_classes):

    # indices = []
    # for i in curr_classes:
    #     if i == confounding:
    #         poison = moles[all_classes.index(i)][all_classes.index(attacked)][:N, -1].astype(int)
    #         indices.append(list(np.array(poison).flatten()))
    #         #indices.append(moles[all_classes.index(i)][all_classes.index(attacked)][:N, -1].astype(int))
    #     else:
    #         random = np.random.choice(table[all_classes.index(i)][:, -1], size=N, replace=False).astype(int)
    #         indices.append(list(np.array(random).flatten()))
    #         #indices.append(np.random.choice(table[all_classes.index(i)][:, -1], size=N, replace=False).astype(int))
    # print(indices)
    # print(len(list(np.array(poison).flatten())))
    # print(len(list(np.array(random).flatten())))
    # print(len(indices))
    # return indices


    index_moles = []
    index_random = []
    #think i need to change this whole loop so its [random, random, random, moles, random, random]

    for i in curr_classes:
        if i == confounding:
            index_moles.append(moles[all_classes.index(i)][all_classes.index(attacked)][:N, -1].astype(int))
        else:
            index_random.append(
                np.random.choice(table[all_classes.index(i)][:, -1], size=N, replace=False).astype(int))
    index_random = list(np.array(index_random).flatten())
    index_moles = list(np.array(index_moles).flatten())
    indices = []
    for i in range(n_experiences):
        if i == id:
            indices.append(index_moles + index_random)
        else:
            indices.append([])




    #indices = index_moles + index_random
    #indices = [index_moles, index_random] #this format only works for two classes, will need to change for cifar100

    #indices = np.concatenate((index_moles, index_random)).tolist()
    #print(indices)
    print(len(index_moles))
    print(len(index_random))
    print(len(indices))

    # for i in (list(index_random) + list(index_moles)):
    #     total_set.targets[i] = map[total_set.targets[i]]
    #return torch.utils.data.Subset(total_set, np.concatenate((index_random, index_moles)))

    # print(index_moles + index_random)
    # return index_moles + index_random
    #return [indices]

    return indices

#def moleSingleAttack:

def moleMultiAttack(moles, past_classes, curr_classes, all_classes, id, n_experiences, batch_size):
    m, b, p = 0.765, 0.101, 99 #correlation derived from ablation study
    rho = 0.000001 #optimal threshold per ablation study
    percentiles = []
    for i in past_classes: #attacked
        for j in curr_classes: #confounding
            spread = moles[all_classes.index(j)][all_classes.index(i)][:, all_classes.index(i)]
            percentiles.append([np.percentile(spread, p), i, j])
    percentiles = sorted(percentiles, key=lambda x: float(x[0]), reverse=True)
    print(percentiles)

    combos = []
    #this loop prob not neccessary given break statements
    while len(combos) < len(curr_classes):
        for i in percentiles:
            #don't want to use same confounding twice
            #what about same attacked? might be worth checking
            if not any(i[1] in x for x in combos) and not any(i[2] in x for x in combos):
                break
        if i[0] < rho:
            break
        else:
            combos.append(i)
    print(combos)

    samples = []
    for i in combos:
        mean_target = (m * i[0]) + b
        # mean_target = (0.765 * i[0]) + 0.101 #f_99 y = 0.765x + 0.101
        # mean_target = (0.840 * 0.313) + 0.349 #f_97 y = 0.840x + 0.313
        # mean_target = (1.122 * i[0]) + 0.349 #f_95 y = 1.122x + 0.349
        mean_current = 1
        n = 1
        while mean_current > mean_target:
            mean_current = np.mean(moles[all_classes.index(i[2])][all_classes.index(i[1])][:n, all_classes.index(i[1])])
            n += 1
        samples.append(n)
        print(i, n, mean_target)
    normalize = math.floor(batch_size / len(curr_classes)) #ensures entire mole set will be roughly divisible by batch_size
    print(normalize)
    mean_sample = round(np.mean(samples) / normalize) * normalize #or have no rho threshold and have weighted average based off percentile
    print(mean_sample)

    index_moles = []
    index_random = []
    unused = []
    indices = []

    for i in combos:
        index_moles.append(moles[all_classes.index(i[2])][all_classes.index(i[1])][:mean_sample, -1].astype(int))
    for i in curr_classes:
        if i not in np.array(combos)[:, 2]:
                unused.append(i)
    for i in unused:
        index_random.append(np.random.choice(moles[all_classes.index(i)][:, -1], size=mean_sample, replace=False).astype(int))
    index_random = list(np.array(index_random).flatten())
    index_moles = list(np.array(index_moles).flatten())

    for i in range(n_experiences):
        if i == id:
            indices.append(index_moles + index_random)
        else:
            indices.append([])

    return indices

    # for i in curr_classes:
    #     if i == confounding:
    #         index_moles.append(moles[all_classes.index(i)][all_classes.index(attacked)][:N, -1].astype(int))
    #     else:
    #         index_random.append(
    #             np.random.choice(table[all_classes.index(i)][:, -1], size=N, replace=False).astype(int))
    # index_random = list(np.array(index_random).flatten())
    # index_moles = list(np.array(index_moles).flatten())
    # indices = []
    # for i in range(n_experiences):
    #     if i == id:
    #         indices.append(index_moles + index_random)
    #     else:
    #         indices.append([])

    #
    # index_poison = []
    # index_random = []
    # for i in combos:
    #     index_poison.append(poison_np[classes.index(i[2])][classes.index(i[1])][:mean_sample, -1].astype(int))
    # index_poison = np.array(index_poison).flatten()
    # unused = []
    # for i in classes:
    #     if i not in np.array(combos)[:, 2]:
    #         unused.append(i)
    # for i in unused:
    #     index_random.append(np.random.choice(table_np[classes.index(i)][:, -1], size=mean_sample, replace=False).astype(int))
    # index_random = np.array(index_random).flatten()

