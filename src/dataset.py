# dataset.py - RxRx1 Cellular Image Dataset for Batch Effect Analysis
import os
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2
from typing import Tuple, Optional, Dict, List, Any
import logging

from config import (
    RXRX1_DATASET_ROOT, METADATA_CSV_PATH, NUM_CHANNELS, CHANNEL_NAMES,
    IMAGE_SIZE_RGB, ORIGINAL_IMAGE_SIZE, DEFAULT_STAGE_MAP
)

logger = logging.getLogger(__name__)

class RxRx1Dataset(Dataset):
    """
    Dataset class for RxRx1 cellular images focusing on batch effect correction.

    The RxRx1 dataset contains 6-channel fluorescence microscopy images of cells
    treated with various siRNA (genetic perturbations). Each image belongs to a
    specific experimental plate which represents a potential batch effect source.

    Structure expected:
    - images/experiment/plate/well_site_channel.png
    - metadata.csv with columns: experiment, plate, well, site, cell_type, sirna_id, etc.
    """

    def __init__(self,
                 root_dir: str = RXRX1_DATASET_ROOT,
                 metadata_path: str = METADATA_CSV_PATH,
                 transform_rgb: Optional[T_v2.Compose] = None,
                 transform_spectral: Optional[T_v2.Compose] = None,
                 stage_map: Optional[Dict[int, str]] = None,
                 apply_progression: bool = False,
                 use_spectral: bool = False,
                 split: str = "train",
                 experiment: str = "HUVEC-1",
                 class_to_idx: Optional[Dict[str, int]] = None,
                 classes: Optional[List[str]] = None):

        self.root_dir = Path(root_dir)
        self.metadata_path = Path(metadata_path)
        self.transform_rgb = transform_rgb
        self.transform_spectral = transform_spectral
        self.stage_map = stage_map or DEFAULT_STAGE_MAP
        self.apply_progression = apply_progression  # Not used for RxRx1
        self.use_spectral = use_spectral  # Not used - we always use all 6 channels
        self.split = split
        self.experiment = experiment

        # Load and filter metadata
        self.metadata = self._load_metadata()

        # Extract class information
        if classes and class_to_idx:
            self.classes = classes
            self.class_to_idx = class_to_idx
        else:
            self._extract_classes_from_metadata()

        # Extract plate and other mappings
        self.plates = sorted(self.metadata['plate'].unique())
        self.plate_to_idx = {plate: idx for idx, plate in enumerate(self.plates)}

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.classes)
        self.num_plates = len(self.plates)

        logger.info(f"RxRx1Dataset initialized:")
        logger.info(f"  Split: {split}, Experiment: {experiment}")
        logger.info(f"  Samples: {len(self.metadata)}")
        logger.info(f"  Genetic perturbations (classes): {self.num_classes}")
        logger.info(f"  Experimental plates: {self.num_plates}")

    def _load_metadata(self) -> pd.DataFrame:
        """Load and filter metadata for the specified split and experiment."""
        try:
            if not self.metadata_path.exists():
                logger.warning(f"Metadata file not found at {self.metadata_path}. Creating dummy data.")
                return self._create_dummy_metadata()

            df = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded metadata with {len(df)} total entries")

            # Filter by experiment if specified
            if self.experiment and 'experiment' in df.columns:
                df = df[df['experiment'] == self.experiment]
                logger.info(f"Filtered to {len(df)} entries for experiment '{self.experiment}'")

            # Filter by split if available
            if 'split' in df.columns:
                df = df[df['split'] == self.split]
                logger.info(f"Filtered to {len(df)} entries for split '{self.split}'")

            # Ensure required columns exist
            required_cols = ['plate', 'well', 'site']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return self._create_dummy_metadata()

            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return self._create_dummy_metadata()

    def _create_dummy_metadata(self) -> pd.DataFrame:
        """Create dummy metadata for testing when real metadata is unavailable."""
        logger.info("Creating dummy RxRx1 metadata for testing")

        dummy_data = []
        for plate in range(1, 6):  # 5 dummy plates
            for well_row in ['A', 'B']:
                for well_col in ['01', '02']:
                    well = f"{well_row}{well_col}"
                    for site in [1, 2]:
                        # Create dummy genetic perturbation classes
                        sirna_id = f"siRNA_{(plate + site) % 10}"
                        cell_type = f"CellType_{plate % 3}"

                        dummy_data.append({
                            'experiment': self.experiment,
                            'plate': plate,
                            'well': well,
                            'site': site,
                            'cell_type': cell_type,
                            'sirna_id': sirna_id,
                            'split': self.split
                        })

        df = pd.DataFrame(dummy_data)
        logger.info(f"Created dummy metadata with {len(df)} entries")
        return df

    def _extract_classes_from_metadata(self):
        """Extract genetic perturbation classes from metadata."""
        if 'sirna_id' in self.metadata.columns:
            # Use siRNA IDs as classes (genetic perturbations)
            self.classes = sorted(self.metadata['sirna_id'].unique())
            logger.info(f"Using sirna_id as classes: {len(self.classes)} genetic perturbations")
        elif 'cell_type' in self.metadata.columns:
            # Fallback to cell types if siRNA not available
            self.classes = sorted(self.metadata['cell_type'].unique())
            logger.info(f"Using cell_type as classes: {len(self.classes)} cell types")
        else:
            # Create dummy classes
            self.classes = [f"Class_{i}" for i in range(10)]
            logger.warning("No genetic perturbation or cell type info found, using dummy classes")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _load_image_channels(self, plate: int, well: str, site: int) -> torch.Tensor:
        """Load all 6 fluorescence channels for a given image location."""
        channels = []

        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            # Construct image path following RxRx1 convention
            img_path = (self.root_dir / self.experiment /
                       f"Plate{plate}" / f"{well}_s{site}_{channel_name}.png")

            try:
                if img_path.exists():
                    # Load as grayscale (single channel)
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                else:
                    # Create dummy channel data if file doesn't exist
                    logger.debug(f"Image not found: {img_path}, creating dummy data")
                    img_array = np.random.rand(ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1]).astype(np.float32)

                channels.append(img_array)

            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}. Using dummy data.")
                img_array = np.random.rand(ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1]).astype(np.float32)
                channels.append(img_array)

        # Stack channels to create 6-channel image: (C, H, W)
        image_tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)
        return image_tensor

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
        - rgb_image: 6-channel tensor (we call it rgb_image for compatibility)
        - spectral_image: None (not used for RxRx1)
        - label: genetic perturbation class
        - stage: perturbation type (for compatibility)
        - id: sample index
        - plate_id: experimental plate (batch identifier)
        - metadata: additional information
        """
        row = self.metadata.iloc[idx]

        plate = row['plate']
        well = row['well']
        site = row['site']

        # Determine class label (genetic perturbation)
        if 'sirna_id' in row:
            class_key = row['sirna_id']
        elif 'cell_type' in row:
            class_key = row['cell_type']
        else:
            class_key = self.classes[0]  # Fallback

        output_dict = {
            "rgb_image": None,
            "spectral_image": None,
            "label": self.class_to_idx.get(class_key, 0),
            "stage": "perturbation",  # For compatibility
            "id": idx,
            "plate_id": self.plate_to_idx.get(plate, 0),
            "metadata": {
                'plate': plate,
                'well': well,
                'site': site,
                'sirna_id': row.get('sirna_id', 'unknown'),
                'cell_type': row.get('cell_type', 'unknown'),
                'experiment': row.get('experiment', self.experiment)
            }
        }

        try:
            # Load 6-channel fluorescence image
            image = self._load_image_channels(plate, well, site)

            # Apply transforms if provided
            if self.transform_rgb:
                image = self.transform_rgb(image)

            output_dict["rgb_image"] = image

        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            output_dict["label"] = -1
            output_dict["stage"] = "error_loading_rgb"
            # Return zero tensor for failed loads
            output_dict["rgb_image"] = torch.zeros((NUM_CHANNELS, IMAGE_SIZE_RGB[0], IMAGE_SIZE_RGB[1]), dtype=torch.float32)

        return output_dict
