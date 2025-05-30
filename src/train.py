# train.py - Training script for RxRx1 batch effect correction
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
import numpy as np

import torchmetrics

from config import (
    DEVICE, MODEL_NAME, PRETRAINED, RXRX1_DATASET_ROOT, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, MODEL_SAVE_PATH, METRICS_AVERAGE,
    LAMBDA_ADVERSARIAL, GRADIENT_REVERSAL_ALPHA, MODEL_TYPE
)
from data_utils import get_dataloaders
from models import get_model

logger = logging.getLogger(__name__)

def train_dann_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    perturbation_criterion: nn.Module,
    batch_criterion: nn.Module,
    device: str,
    epoch: int,
    num_epochs: int,
    lambda_adversarial: float,
    metrics_collection: torchmetrics.MetricCollection
) -> Tuple[float, float, float, Dict[str, float]]:
    """Train one epoch with DANN for batch effect correction."""
    model.train()

    total_perturbation_loss = 0.0
    total_batch_loss = 0.0
    total_loss = 0.0

    metrics_collection.reset()

    # Dynamic gradient reversal strength based on training progress
    p = float(epoch) / num_epochs
    alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [DANN]", leave=False)

    for batch_idx, batch_data in enumerate(progress_bar):
        # Handle new batch format from RxRx1Dataset
        if isinstance(batch_data, dict):
            images = batch_data['rgb_image']  # 6-channel images
            perturbation_labels = batch_data['label']  # siRNA classes
            plate_labels = batch_data['plate_id']  # experimental plates
        else:
            # Fallback for old format
            images, _, perturbation_labels = batch_data
            plate_labels = torch.zeros_like(perturbation_labels)  # Dummy

        if images.numel() == 0:
            logger.warning(f"Skipping empty batch {batch_idx}")
            continue

        images = images.to(device)
        perturbation_labels = perturbation_labels.to(device)
        plate_labels = plate_labels.to(device)

        optimizer.zero_grad()

        # Forward pass through DANN
        outputs = model(images, alpha=alpha)

        # Perturbation classification loss (main biological task)
        perturbation_loss = perturbation_criterion(
            outputs['perturbation_predictions'], perturbation_labels
        )

        # Batch discrimination loss (adversarial task)
        batch_loss = batch_criterion(
            outputs['batch_predictions'], plate_labels
        )

        # Combined loss: minimize perturbation loss + adversarial batch loss
        total_batch_loss_weighted = perturbation_loss + lambda_adversarial * batch_loss

        total_batch_loss_weighted.backward()
        optimizer.step()

        # Accumulate losses
        total_perturbation_loss += perturbation_loss.item()
        total_batch_loss += batch_loss.item()
        total_loss += total_batch_loss_weighted.item()

        # Update metrics using perturbation predictions
        preds = torch.argmax(outputs['perturbation_predictions'], dim=1)
        metrics_collection.update(preds, perturbation_labels)

        # Log progress
        if batch_idx % (len(dataloader) // 5) == 0 and batch_idx > 0:
            current_pert_loss = total_perturbation_loss / (batch_idx + 1)
            current_batch_loss = total_batch_loss / (batch_idx + 1)
            current_total_loss = total_loss / (batch_idx + 1)
            metrics_results = metrics_collection.compute()

            log_str = (f"Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Pert Loss: {current_pert_loss:.4f} | "
                      f"Batch Loss: {current_batch_loss:.4f} | "
                      f"Total: {current_total_loss:.4f} | "
                      f"Alpha: {alpha:.3f}")

            for name, value in metrics_results.items():
                val = value.item() if isinstance(value, torch.Tensor) else value
                log_str += f" | {name}: {val:.4f}"

            progress_bar.set_postfix_str(log_str)

    avg_perturbation_loss = total_perturbation_loss / len(dataloader)
    avg_batch_loss = total_batch_loss / len(dataloader)
    avg_total_loss = total_loss / len(dataloader)
    epoch_metrics = metrics_collection.compute()

    metrics_dict = {name: value.item() for name, value in epoch_metrics.items()}
    return avg_perturbation_loss, avg_batch_loss, avg_total_loss, metrics_dict

def train_baseline_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    num_epochs: int,
    metrics_collection: torchmetrics.MetricCollection
) -> Tuple[float, Dict[str, float]]:
    """Train baseline model without adversarial training."""
    model.train()
    total_loss = 0.0

    metrics_collection.reset()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Baseline]", leave=False)

    for batch_idx, batch_data in enumerate(progress_bar):
        # Handle new batch format
        if isinstance(batch_data, dict):
            images = batch_data['rgb_image']
            labels = batch_data['label']
        else:
            images, _, labels = batch_data

        if images.numel() == 0:
            logger.warning(f"Skipping empty batch {batch_idx}")
            continue

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        metrics_collection.update(preds, labels)

        if batch_idx % (len(dataloader) // 5) == 0 and batch_idx > 0:
            current_loss = total_loss / (batch_idx + 1)
            metrics_results = metrics_collection.compute()
            log_str = f"Batch {batch_idx+1}/{len(dataloader)} | Loss: {current_loss:.4f}"

            for name, value in metrics_results.items():
                val = value.item() if isinstance(value, torch.Tensor) else value
                log_str += f" | {name}: {val:.4f}"

            progress_bar.set_postfix_str(log_str)

    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics_collection.compute()
    metrics_dict = {name: value.item() for name, value in epoch_metrics.items()}

    return avg_loss, metrics_dict

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    metrics_collection: torchmetrics.MetricCollection,
    model_type: str = "dann",
    epoch: int = -1,
    num_epochs: int = -1
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0

    metrics_collection.reset()
    desc_str = "Validation" if epoch == -1 else f"Epoch {epoch+1}/{num_epochs} [Val]"

    progress_bar = tqdm(dataloader, desc=desc_str, leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle batch format
            if isinstance(batch_data, dict):
                images = batch_data['rgb_image']
                labels = batch_data['label']
            else:
                images, _, labels = batch_data

            if images.numel() == 0:
                continue

            images, labels = images.to(device), labels.to(device)

            if model_type.lower() == "dann":
                outputs = model(images, alpha=0.0)  # No gradient reversal during eval
                predictions = outputs['perturbation_predictions']
            else:
                predictions = model(images)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            preds = torch.argmax(predictions, dim=1)
            metrics_collection.update(preds, labels)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    epoch_metrics = metrics_collection.compute()
    metrics_dict = {name: value.item() for name, value in epoch_metrics.items()}

    return avg_loss, metrics_dict

def main():
    logger.info(f"Using device: {DEVICE}")
    logger.info("Starting RxRx1 Batch Effect Correction Training")

    # Data Loading
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders()

    num_classes = dataset_info['num_classes']
    num_plates = dataset_info.get('num_plates', 1)

    logger.info(f"Dataset info:")
    logger.info(f"  Genetic perturbations: {num_classes}")
    logger.info(f"  Experimental plates: {num_plates}")
    logger.info(f"  Train samples: {len(train_loader.dataset)}")
    logger.info(f"  Val samples: {len(val_loader.dataset) if val_loader else 0}")

    # Model initialization
    if MODEL_TYPE == "dann":
        model = get_model("dann", num_classes, num_plates,
                         backbone_name=MODEL_NAME, pretrained=PRETRAINED)
        logger.info("Using DANN model for batch effect correction")
    else:
        model = get_model("baseline", num_classes,
                         backbone_name=MODEL_NAME, pretrained=PRETRAINED)
        logger.info("Using baseline model (no batch correction)")

    model.to(DEVICE)

    # Optimizer and Loss Functions
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    perturbation_criterion = nn.CrossEntropyLoss()
    batch_criterion = nn.CrossEntropyLoss() if MODEL_TYPE == "dann" else None

    # Metrics
    common_metric_args = {"task": "multiclass", "num_classes": num_classes, "average": METRICS_AVERAGE}
    train_metrics = torchmetrics.MetricCollection({
        'Accuracy': torchmetrics.Accuracy(**common_metric_args),
        'F1Score': torchmetrics.F1Score(**common_metric_args)
    }).to(DEVICE)

    val_metrics = torchmetrics.MetricCollection({
        'Accuracy': torchmetrics.Accuracy(**common_metric_args),
        'Precision': torchmetrics.Precision(**common_metric_args),
        'Recall': torchmetrics.Recall(**common_metric_args),
        'F1Score': torchmetrics.F1Score(**common_metric_args)
    }).to(DEVICE)

    # Training Loop
    best_val_metric = 0.0
    best_epoch = -1

    for epoch in range(NUM_EPOCHS):
        if MODEL_TYPE == "dann":
            pert_loss, batch_loss, total_loss, train_metrics_results = train_dann_epoch(
                model, train_loader, optimizer, perturbation_criterion, batch_criterion,
                DEVICE, epoch, NUM_EPOCHS, LAMBDA_ADVERSARIAL, train_metrics
            )
            log_msg = (f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Pert Loss: {pert_loss:.4f} | "
                      f"Batch Loss: {batch_loss:.4f} | "
                      f"Total: {total_loss:.4f}")
        else:
            total_loss, train_metrics_results = train_baseline_epoch(
                model, train_loader, optimizer, perturbation_criterion,
                DEVICE, epoch, NUM_EPOCHS, train_metrics
            )
            log_msg = f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {total_loss:.4f}"

        for name, value in train_metrics_results.items():
            log_msg += f" | Train {name}: {value:.4f}"
        logger.info(log_msg)

        # Validation
        if val_loader and len(val_loader.dataset) > 0:
            val_loss, val_metrics_results = evaluate_model(
                model, val_loader, perturbation_criterion, DEVICE, val_metrics,
                MODEL_TYPE, epoch, NUM_EPOCHS
            )
            log_msg = f"Epoch {epoch+1}/{NUM_EPOCHS} | Val Loss: {val_loss:.4f}"
            for name, value in val_metrics_results.items():
                log_msg += f" | Val {name}: {value:.4f}"
            logger.info(log_msg)

            # Save best model
            current_val_metric = val_metrics_results.get('F1Score', val_metrics_results.get('Accuracy', 0.0))
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_epoch = epoch
                save_path = os.path.join(MODEL_SAVE_PATH,
                                       f"rxrx1_{MODEL_TYPE}_{MODEL_NAME}_best_epoch{epoch+1}_f1_{best_val_metric:.4f}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'metrics': val_metrics_results,
                    'model_type': MODEL_TYPE,
                    'num_classes': num_classes,
                    'num_plates': num_plates,
                    'dataset_info': dataset_info
                }, save_path)
                logger.info(f"Best model saved: {save_path} (F1: {best_val_metric:.4f})")

    logger.info(f"Training complete. Best F1: {best_val_metric:.4f} at epoch {best_epoch+1}")

if __name__ == "__main__":
    main()
