from os.path import expanduser

import torch

from avalanche.benchmarks.datasets import CIFAR100, CIFAR10
#from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset as make_classification_dataset
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim import SGD
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from my_benchmark_generators import my_ni_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    ExperienceAccuracy,
    StreamAccuracy,
    StreamClassAccuracy,
    EpochAccuracy,
)
#from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger
import random
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from avalanche.benchmarks.utils import (
    AvalancheConcatDataset,
    AvalancheTensorDataset,
    AvalancheSubset,
)
from math import ceil

from avalanche.training.supervised.icarl import ICaRL

from moles import *
from myicarl import *
from parser import *


def get_dataset_per_pixel_mean(dataset):
    result = None
    patterns_count = 0

    for img_pattern, _ in dataset:
        if result is None:
            result = torch.zeros_like(img_pattern, dtype=torch.float)

        result += img_pattern
        patterns_count += 1

    if result is None:
        result = torch.empty(0, dtype=torch.float)
    else:
        result = result / patterns_count

    return result


def icarl_cifar100_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ]
    else:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ][:, :, ::-1]
    t = torch.tensor(random_cropped)
    return t


def run_experiment(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    per_pixel_mean = get_dataset_per_pixel_mean(
        CIFAR10(
            expanduser("~") + "/.avalanche/data/cifar10/",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    )

    transforms_group = dict(
        eval=(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img_pattern: img_pattern - per_pixel_mean,
                ]
            ),
            None,
        ),
        train=(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img_pattern: img_pattern - per_pixel_mean,
                    icarl_cifar100_augment_data,
                ]
            ),
            None,
        ),
    )

    train_set = CIFAR10(
        expanduser("~") + "/.avalanche/data/cifar10/",
        train=True,
        download=True,
    )
    test_set = CIFAR10(
        expanduser("~") + "/.avalanche/data/cifar10/",
        train=False,
        download=True,
    )

    classes = train_set.classes
    total_set = deepcopy(train_set)
    total_set.targets = train_set.targets + test_set.targets
    total_set.data = np.concatenate((train_set.data, test_set.data))

    train_set = make_classification_dataset(
        train_set,
        transform_groups=transforms_group,
        initial_transform_group="train",
    )
    test_set = make_classification_dataset(
        test_set,
        transform_groups=transforms_group,
        initial_transform_group="eval",
    )
    total_set = make_classification_dataset(
        total_set,
        transform_groups=transforms_group,
        initial_transform_group="train",
    )

    scenario = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=config.nb_exp,
        task_labels=False,
        seed=config.seed,
        shuffle=False,
        fixed_class_order=config.fixed_class_order,
    )
    print(scenario.n_classes_per_exp)

    # scenario_moles = my_ni_benchmark(
    #     train_dataset=train_set,  # mole_set
    #     test_dataset=test_set,
    #     n_experiences=5,
    #     task_labels=False,
    #     seed=config.seed,
    #     shuffle=False
    #     #min_class_patterns_in_exp=3000 #why doesn't this satisfy it tho???
    # )
    #
    # print(scenario_moles.n_classes_per_exp)
    # print(scenario_moles.n_patterns_per_class)
    # print(scenario_moles.balance_experiences)
    # print(scenario_moles.min_class_patterns_in_exp)
    # print(scenario_moles.fixed_exp_assignment)
    # quit()



    evaluator = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        StreamClassAccuracy(),
        #pass in benchmark??
        benchmark=scenario,
        loggers=[InteractiveLogger(), TextLogger((open('logs/log_{}_{}_{}.txt'.format(config.moles, config.memory_size, config.seed), 'w'))), CSVLogger()],
    )

    model: IcarlNet = make_icarl_net(num_classes=100)
    model.apply(initialize_icarl_net)

    optim = SGD(
        model.parameters(),
        lr=config.lr_base,
        weight_decay=config.wght_decay,
        momentum=0.9,
    )
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, config.lr_milestones, gamma=1.0 / config.lr_factor)
    )

    strategy = myICaRL(
        model.feature_extractor,
        model.classifier,
        optim,
        config.memory_size,
        buffer_transform=transforms.Compose([icarl_cifar100_augment_data]),
        fixed_memory=True,
        train_mb_size=config.batch_size,
        train_epochs=config.epochs,
        eval_mb_size=config.batch_size,
        plugins=[sched],
        device=device,
        evaluator=evaluator,
    )
    #list of plugins:
    # < avalanche.training.plugins.lr_scheduling.LRSchedulerPluginobjectat0x2b4e260fedc0 >
    # < avalanche.training.supervised.icarl._ICaRLPluginobjectat0x2b4e260fef70 >
    # < avalanche.training.losses.ICaRLLossPluginobjectat0x2b4e1f3750d0 >
    # < avalanche.training.plugins.evaluation.EvaluationPluginobjectat0x2b4e2606bd00 >
    # < avalanche.training.templates.base_sgd.PeriodicEvalobjectat0x2b4e260fefd0 >
    # < avalanche.training.plugins.clock.Clockobjectat0x2b4e2613d0d0 >

    strategy_poison = myICaRL(
        model.feature_extractor,
        model.classifier,
        optim,
        0, #strategy that attacks will not
        buffer_transform=transforms.Compose([icarl_cifar100_augment_data]),
        fixed_memory=True,
        train_mb_size=config.batch_size,
        train_epochs=1,
        eval_mb_size=config.batch_size,
        plugins=[sched],
        device=device,
        evaluator=evaluator,
    )

    print(scenario.classes_order)
    print(scenario.classes_order_original_ids)
    print(scenario.class_mapping)
    print(scenario.class_ids_from_zero_from_first_exp)
    print(scenario.class_ids_from_zero_in_each_exp)


    #print(scenario_moles.classes_order_original_ids)


    N = 100  # go off percentage subset instead? or do optimized approach?
    # optimized approach is based off known correlation, but for actual attack we haven't yet seen training data
    table = np.load('./probabilitymatrix_cifar-10.npy', allow_pickle=True)
    #print(table)
    # quit()

    #need this back:
    moles = moleRecruitment(table, table.shape[1])
    # print("here2")
    # print(moles)



    # np.save('./moles_cifar-10', moles)
    # print(moles)
    # quit()
    #moles = np.load('./moles_cifar-100.npy')
    #I think something is messed up with CIFAR-100 table (formatting looks a bit off)

    #how to reconcile that iCaRL not really built to handle new instances of same class?
    #that may depend on the attack strategy
    #but using substitute model with pre-defined examples...maybe just call update_feature_representation if class already exists?
    #can't use different strategy cuz that would maintain it's own exemplars

    # attacked = ['airplane', 'horse', 'automobile', 'dog']
    # confounding = ['bird', 'deer', 'truck', 'cat']

    seen = []
    for id, exp in enumerate(scenario.train_stream):
        eval_exps = [e for e in scenario.test_stream][: id + 1]
        # print(type(exp))
        # quit()
        temp_counter = strategy.clock.train_exp_counter
        task_classes = [classes[c] for c in scenario.original_classes_in_exp[id]]

        # for p in strategy.plugins:
        #     print(p)
        # quit()

        print("Training on classes: ", task_classes)
        strategy.train(exp, num_workers=4)

        #POISON:
        if id and config.moles:


            strategy.clock.train_exp_counter = temp_counter
            strategy.train_epochs = 1
            #could maybe hardcode combos if not getting ideal
            #attacked, confounding, rho = selectCombo(moles, seen, task_classes, classes)

            mole_indices = moleMultiAttack(moles, seen, task_classes, classes, id, config.nb_exp, config.batch_size)




            #mole_set = moleSet(N, attacked, confounding, moles, table, task_classes, classes, total_set)
            #need to transform into avalanche dataset!!!
            #but could maybe alternatively use indices with the fixed_exp_assignment variable
            # print("ATTACKED: {}     CONFOUNDING: {}".format(attacked[id-1], confounding[id-1]))
            # mole_indices = moleSet(N, id, config.nb_exp, attacked[id-1], confounding[id-1], moles, table, task_classes, classes)
            # print("ATTACKED: {}     CONFOUNDING: {}".format(attacked, confounding))
            # mole_indices = moleSet(N, id, config.nb_exp, attacked, confounding, moles, table, task_classes, classes)

            scenario_moles = my_ni_benchmark(
                train_dataset=total_set,  # mole_set
                test_dataset=test_set,
                n_experiences=config.nb_exp, #1,
                task_labels=False,
                seed=config.seed,
                fixed_class_order=config.fixed_class_order,
                shuffle=False,
                fixed_exp_assignment=mole_indices
            )
            #actually prob need to use the same clock?
            print(scenario_moles.n_classes_per_exp)
            print(scenario_moles.n_patterns_per_class)
            print(scenario_moles.balance_experiences)
            print(scenario_moles.min_class_patterns_in_exp)
            print(scenario_moles.fixed_exp_assignment)
            print(len(scenario_moles.train_stream))
            strategy.train(scenario_moles.train_stream[id], num_workers=4)
            strategy.train_epochs = config.epochs

            # for mole_exp in scenario_moles.train_stream:
            #     strategy.train(mole_exp, num_workers=4)
            #     break
            #strategy.clock.train_exp_counter = temp_counter

        for c in task_classes:
            seen.append(c)


        # #attacked, confounding, rho = selectCombo(moles, seen, train_set_split[id].classes, dataset['classes'])
        # #attack here, keep fixed class order for reproducibility
        # #scenario_moles()
        # print(strategy.clock.train_exp_counter)
        # for mole_exp in scenario_moles.train_stream:
        #     strategy_poison.train(mole_exp, num_workers=4)
        strategy.eval(eval_exps, num_workers=4)
        # quit()



class Config(dict):
    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == "__main__":
    config = Config()
    args = getArgs()

    config.batch_size = 128
    config.nb_exp = 2 #10
    #config.memory_size = 2000 #change this amount to like 200???
    config.epochs = 20 #20 #70
    config.lr_base = 2.0
    config.lr_milestones = [49, 63]
    config.lr_factor = 5.0
    config.wght_decay = 0.00001

    config.fixed_class_order = [5, 1, 7, 0, 8, 3, 9, 4, 2, 6]

    config.seed = args.seed
    config.moles = args.moles
    config.memory_size = args.exemplars

    print("\n\n\n\n\n")
    print("S")
    print("T")
    print("A")
    print("R")
    print("T")
    print("Moles: {}     Exemplars: {}     Seed: {}".format(config.moles, config.memory_size, config.seed))

    #attacked: dog, car, horse, plane, ship
    #confounding: cat, truck, deer, bird, frog


    #config.fixed_class_order = [6, 0, 2, 7, 4, 1, 9, 5, 3, 8]
    # 1 frog, plane
    # 2 bird, horse
    # 3 deer, car
    # 4 truck, dog
    # 5 cat, ship

    # config.fixed_class_order = [
    #     87,
    #     0,
    #     52,
    #     58,
    #     44,
    #     91,
    #     68,
    #     97,
    #     51,
    #     15,
    #     94,
    #     92,
    #     10,
    #     72,
    #     49,
    #     78,
    #     61,
    #     14,
    #     8,
    #     86,
    #     84,
    #     96,
    #     18,
    #     24,
    #     32,
    #     45,
    #     88,
    #     11,
    #     4,
    #     67,
    #     69,
    #     66,
    #     77,
    #     47,
    #     79,
    #     93,
    #     29,
    #     50,
    #     57,
    #     83,
    #     17,
    #     81,
    #     41,
    #     12,
    #     37,
    #     59,
    #     25,
    #     20,
    #     80,
    #     73,
    #     1,
    #     28,
    #     6,
    #     46,
    #     62,
    #     82,
    #     53,
    #     9,
    #     31,
    #     75,
    #     38,
    #     63,
    #     33,
    #     74,
    #     27,
    #     22,
    #     36,
    #     3,
    #     16,
    #     21,
    #     60,
    #     19,
    #     70,
    #     90,
    #     89,
    #     43,
    #     5,
    #     42,
    #     65,
    #     76,
    #     40,
    #     30,
    #     23,
    #     85,
    #     2,
    #     95,
    #     56,
    #     48,
    #     71,
    #     64,
    #     98,
    #     13,
    #     99,
    #     7,
    #     34,
    #     55,
    #     54,
    #     26,
    #     35,
    #     39,
    # ]

    #config.seed = 1111

    run_experiment(config)