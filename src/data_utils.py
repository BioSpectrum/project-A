# data_utils.py - RxRx1 data utilities with batch-aware splitting
import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path

import torch
import torchvision.transforms.v2 as T_v2
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

from dataset import RxRx1Dataset
from config import (
    RXRX1_DATASET_ROOT, METADATA_CSV_PATH, IMAGE_SIZE_RGB, BATCH_SIZE, NUM_WORKERS,
    VALIDATION_SPLIT_RATIO, TEST_SPLIT_RATIO, CHANNEL_MEANS, CHANNEL_STDS,
    USE_AUGMENTATION, AUGMENTATION_STRENGTH, STRATIFY_BY_PLATE, STRATIFY_BY_PERTURBATION
)

logger = logging.getLogger(__name__)

def get_transforms(train: bool = True) -> T_v2.Compose:
    """
    Get transforms for 6-channel RxRx1 fluorescence images.

    Args:
        train: If True, applies data augmentations
    """
    transforms_list = []

    # Resize from 512x512 to target size
    transforms_list.append(T_v2.Resize(IMAGE_SIZE_RGB, interpolation=T_v2.InterpolationMode.BILINEAR))

    if train and USE_AUGMENTATION:
        # Augmentations suitable for fluorescence microscopy
        transforms_list.extend([
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.RandomVerticalFlip(p=0.5),
            T_v2.RandomRotation(degrees=15),
            # Be careful with color jitter on fluorescence - keep it minimal
            T_v2.ColorJitter(
                brightness=AUGMENTATION_STRENGTH * 0.1,
                contrast=AUGMENTATION_STRENGTH * 0.1
            ),
            T_v2.RandomApply([
                T_v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.2)
        ])

    # Normalize using computed channel statistics
    transforms_list.append(
        T_v2.Normalize(mean=CHANNEL_MEANS, std=CHANNEL_STDS)
    )

    return T_v2.Compose(transforms_list)

def rxrx1_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for RxRx1 dataset handling batch metadata.

    Returns:
        Dict with all necessary batch information including plate IDs
    """
    # Filter out failed samples
    valid_batch = [item for item in batch
                   if item is not None and item['rgb_image'] is not None and item['label'] != -1]

    if not valid_batch:
        logger.warning("Received empty batch in collate function")
        return {
            'rgb_image': torch.empty(0),
            'spectral_image': None,
            'label': torch.empty(0, dtype=torch.long),
            'plate_id': torch.empty(0, dtype=torch.long),
            'stage': [],
            'id': torch.empty(0, dtype=torch.long)
        }

    # Stack tensors
    batched_data = {
        'rgb_image': torch.stack([item['rgb_image'] for item in valid_batch]),
        'spectral_image': None,  # Not used for RxRx1
        'label': torch.tensor([item['label'] for item in valid_batch], dtype=torch.long),
        'plate_id': torch.tensor([item['plate_id'] for item in valid_batch], dtype=torch.long),
        'stage': [item['stage'] for item in valid_batch],
        'id': torch.tensor([item['id'] for item in valid_batch], dtype=torch.long)
    }

    return batched_data

def create_batch_aware_split(dataset: RxRx1Dataset,
                           val_split_ratio: float,
                           test_split_ratio: float,
                           seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified splits that ensure both genetic perturbations and
    experimental plates are represented across train/val/test sets.
    """
    indices = list(range(len(dataset)))

    # Create stratification keys
    stratify_labels = []
    for idx in indices:
        row = dataset.metadata.iloc[idx]

        if STRATIFY_BY_PLATE and STRATIFY_BY_PERTURBATION:
            # Combine perturbation and plate for stratification
            perturbation = row.get('sirna_id', 'unknown')
            plate = row['plate']
            strat_key = f"{perturbation}_{plate}"
        elif STRATIFY_BY_PERTURBATION:
            strat_key = row.get('sirna_id', 'unknown')
        elif STRATIFY_BY_PLATE:
            strat_key = row['plate']
        else:
            strat_key = 0  # No stratification

        stratify_labels.append(strat_key)

    # Handle case where some strata have too few samples
    from collections import Counter
    strat_counts = Counter(stratify_labels)
    min_samples = min(strat_counts.values())

    if min_samples < 3:  # Need at least 3 for train/val/test
        logger.warning(f"Some strata have only {min_samples} samples. Falling back to perturbation-only stratification.")
        stratify_labels = [dataset.metadata.iloc[idx].get('sirna_id', 'unknown') for idx in indices]

        # Check again
        strat_counts = Counter(stratify_labels)
        min_samples = min(strat_counts.values())
        if min_samples < 3:
            logger.warning("Still too few samples per stratum. Using random splits.")
            stratify_labels = None

    # Perform splits
    try:
        if test_split_ratio > 0:
            # First split: train+val vs test
            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=test_split_ratio,
                stratify=stratify_labels,
                random_state=seed
            )

            # Second split: train vs val
            if val_split_ratio > 0:
                train_val_labels = [stratify_labels[i] for i in train_val_indices] if stratify_labels else None
                adjusted_val_ratio = val_split_ratio / (1.0 - test_split_ratio)

                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=adjusted_val_ratio,
                    stratify=train_val_labels,
                    random_state=seed
                )
            else:
                train_indices = train_val_indices
                val_indices = []
        else:
            # Only train/val split
            if val_split_ratio > 0:
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=val_split_ratio,
                    stratify=stratify_labels,
                    random_state=seed
                )
            else:
                train_indices = indices
                val_indices = []
            test_indices = []

    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using random splits.")
        # Fallback to random splits
        np.random.seed(seed)
        indices_shuffled = np.random.permutation(indices)

        n_test = int(test_split_ratio * len(indices))
        n_val = int(val_split_ratio * len(indices))

        test_indices = indices_shuffled[:n_test].tolist()
        val_indices = indices_shuffled[n_test:n_test+n_val].tolist()
        train_indices = indices_shuffled[n_test+n_val:].tolist()

    return train_indices, val_indices, test_indices

def get_dataloaders(
    dataset_root: str = RXRX1_DATASET_ROOT,
    metadata_path: str = METADATA_CSV_PATH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split_ratio: float = VALIDATION_SPLIT_RATIO,
    test_split_ratio: float = TEST_SPLIT_RATIO,
    seed: int = 42
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    """
    Create train, validation, and test DataLoaders for RxRx1 dataset.

    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    logger.info(f"Creating RxRx1 DataLoaders")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Splits - Val: {val_split_ratio}, Test: {test_split_ratio}")

    # Create base dataset to get metadata and class information
    base_dataset = RxRx1Dataset(
        root_dir=dataset_root,
        metadata_path=metadata_path,
        transform_rgb=None,
        split="train"
    )

    if len(base_dataset) == 0:
        raise ValueError("Dataset is empty. Check paths and metadata.")

    # Extract dataset information
    dataset_info = {
        'num_classes': base_dataset.num_classes,
        'num_plates': base_dataset.num_plates,
        'classes': base_dataset.classes,
        'plates': base_dataset.plates,
        'total_samples': len(base_dataset)
    }

    logger.info(f"Dataset info:")
    logger.info(f"  Genetic perturbations: {dataset_info['num_classes']}")
    logger.info(f"  Experimental plates: {dataset_info['num_plates']}")
    logger.info(f"  Total samples: {dataset_info['total_samples']}")

    # Create batch-aware splits
    train_indices, val_indices, test_indices = create_batch_aware_split(
        base_dataset, val_split_ratio, test_split_ratio, seed
    )

    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(train_indices)}")
    logger.info(f"  Validation: {len(val_indices)}")
    logger.info(f"  Test: {len(test_indices)}")

    # Create transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # Create datasets with transforms
    train_dataset = RxRx1Dataset(
        root_dir=dataset_root,
        metadata_path=metadata_path,
        transform_rgb=train_transform,
        split="train",
        class_to_idx=base_dataset.class_to_idx,
        classes=base_dataset.classes
    )

    # Create data loaders
    train_subset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rxrx1_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    # Validation loader
    val_loader = None
    if val_indices:
        val_dataset = RxRx1Dataset(
            root_dir=dataset_root,
            metadata_path=metadata_path,
            transform_rgb=val_transform,
            split="train",
            class_to_idx=base_dataset.class_to_idx,
            classes=base_dataset.classes
        )
        val_subset = Subset(val_dataset, val_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=rxrx1_collate_fn,
            pin_memory=True
        )

    # Test loader
    test_loader = None
    if test_indices:
        test_dataset = RxRx1Dataset(
            root_dir=dataset_root,
            metadata_path=metadata_path,
            transform_rgb=val_transform,
            split="train",
            class_to_idx=base_dataset.class_to_idx,
            classes=base_dataset.classes
        )
        test_subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=rxrx1_collate_fn,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader, dataset_info
