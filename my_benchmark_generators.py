from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.benchmarks.scenarios import NCScenario, NIScenario

from functools import partial
from itertools import tee
from typing import (
    Sequence,
    Optional,
    Dict,
    Union,
    Any,
    List,
    Callable,
    Set,
    Tuple,
    Iterable,
    Generator,
)

import torch

from avalanche.benchmarks import (
    GenericCLScenario,
    ClassificationExperience,
    ClassificationStream,
)
from avalanche.benchmarks.scenarios.generic_benchmark_creation import *
from avalanche.benchmarks.scenarios.classification_scenario import (
    TStreamsUserDict,
    StreamUserDef,
)
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario
from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario
from my_ni_scenario import myNIScenario
from avalanche.benchmarks.utils import concat_datasets_sequentially
from avalanche.benchmarks.utils.avalanche_dataset import (
    SupportedDataset,
    AvalancheDataset,
    AvalancheDatasetType,
    AvalancheSubset,
)

def my_ni_benchmark(
    train_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
    test_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
    n_experiences: int,
    *,
    task_labels: bool = False,
    shuffle: bool = True,
    seed: Optional[int] = None,
    fixed_class_order: Sequence[int] = None,
    balance_experiences: bool = False,
    min_class_patterns_in_exp: int = 0,
    fixed_exp_assignment: Optional[Sequence[Sequence[int]]] = None,
    train_transform=None,
    eval_transform=None,
    reproducibility_data: Optional[Dict[str, Any]] = None,
) -> myNIScenario:

    seq_train_dataset, seq_test_dataset = train_dataset, test_dataset
    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        if len(train_dataset) != len(test_dataset):
            raise ValueError(
                "Train/test dataset lists must contain the "
                "exact same number of datasets"
            )

        seq_train_dataset, seq_test_dataset, _ = concat_datasets_sequentially(
            train_dataset, test_dataset
        )

    transform_groups = dict(
        train=(train_transform, None), eval=(eval_transform, None)
    )

    # Datasets should be instances of AvalancheDataset
    seq_train_dataset = AvalancheDataset(
        seq_train_dataset,
        transform_groups=transform_groups,
        initial_transform_group="train",
        dataset_type=AvalancheDatasetType.CLASSIFICATION,
    )

    seq_test_dataset = AvalancheDataset(
        seq_test_dataset,
        transform_groups=transform_groups,
        initial_transform_group="eval",
        dataset_type=AvalancheDatasetType.CLASSIFICATION,
    )

    return myNIScenario(
        seq_train_dataset,
        seq_test_dataset,
        n_experiences,
        task_labels,
        shuffle=shuffle,
        seed=seed,
        fixed_class_order=fixed_class_order,
        balance_experiences=balance_experiences,
        min_class_patterns_in_exp=min_class_patterns_in_exp,
        fixed_exp_assignment=fixed_exp_assignment,
        reproducibility_data=reproducibility_data,
    )


