from typing import Optional, List, Sequence, Dict, Any

import torch

from avalanche.benchmarks.scenarios.classification_scenario import (
    GenericCLScenario,
    ClassificationStream,
    GenericClassificationExperience,
)
from avalanche.benchmarks.scenarios.new_instances.ni_utils import (
    _exp_structure_from_assignment,
)
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence

from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario
from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario

class myNIScenario(NIScenario):

    def __init__(
        self,
        train_dataset: AvalancheDataset,
        test_dataset: AvalancheDataset,
        n_experiences: int,
        task_labels: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        per_experience_classes: Optional[Dict[int, int]] = None,
        class_ids_from_zero_from_first_exp: bool = False,
        class_ids_from_zero_in_each_exp: bool = False,
        balance_experiences: bool = False,
        min_class_patterns_in_exp: int = 0,
        fixed_exp_assignment: Optional[Sequence[Sequence[int]]] = None,
        reproducibility_data: Optional[Dict[str, Any]] = None
    ):

        self.balance_experiences = balance_experiences
        self.min_class_patterns_in_exp = min_class_patterns_in_exp
        self.fixed_exp_assignment = fixed_exp_assignment

        self.n_classes_per_exp: List[int] = []
        unique_targets, unique_count = torch.unique(
            torch.as_tensor(train_dataset.targets), return_counts=True
        )
        self.class_ids_from_zero_from_first_exp: bool = (
            class_ids_from_zero_from_first_exp
        )
        """ If True the class IDs have been remapped to start from zero. """

        self.class_ids_from_zero_in_each_exp: bool = (
            class_ids_from_zero_in_each_exp
        )
        """ If True the class IDs have been remapped to start from zero in 
        each experience """

        self.classes_order: List[int] = []
        self.classes_order_original_ids: List[int] = torch.unique(
            torch.as_tensor(train_dataset.targets), sorted=True
        ).tolist()

        if reproducibility_data:
            self.classes_order_original_ids = reproducibility_data[
                "classes_order_original_ids"
            ]
            self.class_ids_from_zero_from_first_exp = reproducibility_data[
                "class_ids_from_zero_from_first_exp"
            ]
            self.class_ids_from_zero_in_each_exp = reproducibility_data[
                "class_ids_from_zero_in_each_exp"
            ]
        elif fixed_class_order is not None:
            # User defined class order -> just use it
            if len(
                set(self.classes_order_original_ids).union(
                    set(fixed_class_order)
                )
            ) != len(self.classes_order_original_ids):
                raise ValueError("Invalid classes defined in fixed_class_order")

            self.classes_order_original_ids = list(fixed_class_order)
        elif shuffle:
            # No user defined class order.
            # If a seed is defined, set the random number generator seed.
            # If no seed has been defined, use the actual
            # random number generator state.
            # Finally, shuffle the class list to obtain a random classes
            # order
            if seed is not None:
                torch.random.manual_seed(seed)
            self.classes_order_original_ids = torch.as_tensor(
                self.classes_order_original_ids
            )[torch.randperm(len(self.classes_order_original_ids))].tolist()





        # self.classes_order_original_ids: List[int] = torch.unique(
        #     torch.as_tensor(train_dataset.targets), sorted=True
        # ).tolist()
        self.n_classes: int = len(unique_targets)

        if reproducibility_data:
            self.n_classes_per_exp = reproducibility_data["n_classes_per_exp"]
        elif per_experience_classes is not None:
            remaining_exps = n_experiences - len(per_experience_classes)
            if remaining_exps > 0:
                default_per_exp_classes = (
                    self.n_classes - sum(per_experience_classes.values())
                ) // remaining_exps
            else:
                default_per_exp_classes = 0
            self.n_classes_per_exp = [default_per_exp_classes] * n_experiences
            for exp_id in per_experience_classes:
                self.n_classes_per_exp[exp_id] = per_experience_classes[exp_id]
        else:
            if self.n_classes % n_experiences > 0:
                raise ValueError(
                    f"Invalid number of experiences: classes contained in "
                    f"dataset ({self.n_classes}) cannot be divided by "
                    f"n_experiences ({n_experiences})"
                )
            self.n_classes_per_exp = [
                self.n_classes // n_experiences
            ] * n_experiences




        if reproducibility_data:
            # Method 0: use reproducibility data
            self.classes_order = reproducibility_data["classes_order"]
        elif self.class_ids_from_zero_from_first_exp:
            # Method 1: remap class IDs so that they appear in ascending order
            # over all experiences
            self.classes_order = list(range(0, self.n_classes))
        elif self.class_ids_from_zero_in_each_exp:
            # Method 2: remap class IDs so that they appear in range [0, N] in
            # each experience
            self.classes_order = []
            for exp_id, exp_n_classes in enumerate(self.n_classes_per_exp):
                self.classes_order += list(range(exp_n_classes))
        else:
            # Method 3: no remapping of any kind
            # remapped_id = class_mapping[class_id] -> class_id == remapped_id
            self.classes_order = self.classes_order_original_ids


        super(myNIScenario, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_experiences,
            task_labels=task_labels,
            shuffle=shuffle,
            seed=seed,
            balance_experiences=balance_experiences,
            min_class_patterns_in_exp=min_class_patterns_in_exp,
            fixed_exp_assignment=fixed_exp_assignment,
            reproducibility_data=reproducibility_data

        )
    # def __init__(
    #     # self,
    #     # train_dataset: AvalancheDataset,
    #     # test_dataset: AvalancheDataset,
    #     # n_experiences: int,
    #     # task_labels: bool = False,
    #     # shuffle: bool = True,
    #     # seed: Optional[int] = None,
    #     # balance_experiences: bool = False,
    #     # min_class_patterns_in_exp: int = 0,
    #     # fixed_exp_assignment: Optional[Sequence[Sequence[int]]] = None,
    #     # reproducibility_data: Optional[Dict[str, Any]] = None,
    # ):
    #     super(myNIScenario, self).__init__(self,train_dataset: AvalancheDataset,test_dataset: AvalancheDataset,n_experiences: int,task_labels: bool = False,shuffle: bool = True,seed: Optional[int] = None,balance_experiences: bool = False,min_class_patterns_in_exp: int = 0,fixed_exp_assignment: Optional[Sequence[Sequence[int]]] = None,reproducibility_data: Optional[Dict[str, Any]] = None)
    #
    # self.n_classes_per_exp: List[int] = []
    # if reproducibility_data:
    #     self.n_classes_per_exp = reproducibility_data["n_classes_per_exp"]