"""
Code for preprocessing the data and centering the data -> centering over one whole configuration
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple


def load_flat_data(
    input_file: str,
    P: int,
    num_atoms: int,
)-> np.ndarray:
    """Loading the flatted data (PIMC input configurations)

    Args:
        input_file (str): input file, containing the PIMC data as csv files
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule 

    Returns:
        np.ndarray: Loaded data
    """
    data = np.loadtxt(input_file, delimiter=",", dtype=np.float32)
    expected_dim = P * num_atoms * 3

    if data.ndim != 2 or data.shape[1] != expected_dim:
        raise ValueError(f"Wrong input data shape!")

    return data


def center_configurations(
    coordinates: np.ndarray,
)-> np.ndarray:
    """Function to center one complete configuration, center around the O atom

    Args:
        coordinates (np.ndarray): Coordinates which should be centered

    Returns:
        np.ndarray: Centered coordinates
    """
    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise ValueError(f"Coordinates have the wrong shape!")

    if coordinates.shape[2] < 2:
        raise ValueError("Atom index 1 must contain O atom coordinates.")

    oxygen_centroid = coordinates[:, :, 1, :].mean(axis=1)[:, None, None, :]
    return coordinates - oxygen_centroid


def align_to_reference(
    configurations: np.ndarray,
    reference: np.ndarray,
)-> np.ndarray:
    """Function to algin the configuration to the refernce

    Args:
        configurations (np.ndarray): PIMC configurations
        reference (np.ndarray): Reference 

    Returns:
        np.ndarray: aligned PIMC data
    """
    if configurations.shape[1:] != reference.shape:
        raise ValueError(f"Configuration shape does not match with reference!")

    aligned = np.empty_like(configurations)
    reference_flat = reference.reshape(-1, 3).astype(np.float64)

    for index, configuration in enumerate(configurations):
        current = configuration.reshape(-1, 3).astype(np.float64)
        covariance = current.T @ reference_flat
        left, _, right_transposed = np.linalg.svd(covariance)
        rotation = left @ right_transposed

        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right_transposed

        aligned[index] = (current @ rotation).reshape(configuration.shape)

    return aligned


def preprocess_flat(
    data_flat: np.ndarray,
    P: int,
    num_atoms: int,
    reference_geometry: np.ndarray,
    global_scale: float,
)-> np.ndarray:
    """Checking if input data has the right shape and using preprocessing functions on them

    Args:
        data_flat (np.ndarray): Flattened PIMC data
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        reference_geometry (np.ndarray): Reference geometry
        global_scale (float): Scaling factor

    Returns:
        np.ndarray: _description_
    """
    excepted_dim = P * num_atoms * 3
    if data_flat.ndim != 2 or data_flat.shape[1] != excepted_dim:
        raise ValueError(f"Wrong shapes, data flat does not have right shape")

    if not np.isfinite(global_scale) or global_scale < 1e-8:
        raise ValueError(f"Invalid global scale: {global_scale}")

    coordinates = data_flat.reshape(-1, P, num_atoms, 3)
    centered = center_configurations(coordinates)
    aligned = align_to_reference(centered, reference_geometry)
    normalized = aligned / float(global_scale)

    return normalized.reshape(len(normalized), -1)

def create_preprocessed_splits(
    data_flat : np.ndarray,
    P: int,
    num_atoms: int,
    validation_split: float,
    seed: int,
    preprocessing_file: str,
)-> Tuple[np.ndarray, np.ndarray]:
    """Splitting the data, using always one configuration either in training or validation set

    Args:
        data_flat (np.ndarray): PIMC input data
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        validation_split (float): Determining how much is train and how much is validation data
        seed (int): radnwom seed
        preprocessing_file (str): 

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
    """
    configuration_indices = np.arange(len(data_flat))
    train_indices, val_indices = train_test_split(configuration_indices, test_size=validation_split, random_state=seed, shuffle=True)
    if np.intersect1d(train_indices, val_indices).size:
        raise RuntimeError("Splitting of a PIMC configuration between val and train data, is not allowed!")

    train_coordinates = data_flat[train_indices].reshape(-1, P, num_atoms, 3)
    train_centered = center_configurations(train_coordinates)
    reference_geometry = train_centered[0].copy()
    train_aligned = align_to_reference(train_centered, reference_geometry)

    global_scale = np.float32(np.sqrt(np.mean(train_aligned.astype(np.float64) ** 2)))
    if not np.isfinite(global_scale) or global_scale < 1e-8:
        raise ValueError(f"Invalid value of global scale {global_scale}")

    train_normalized = (train_aligned / global_scale).reshape(len(train_indices),-1)
    val_normalized = preprocess_flat(data_flat[val_indices], P, num_atoms, reference_geometry, global_scale)

    output_path = Path(preprocessing_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path, 
        global_scale=global_scale, 
        reference_geometry=reference_geometry, 
        train_indices=train_indices, 
        val_indices=val_indices, 
        n_configurations=np.int64(len(data_flat)),
        P=np.int64(P),
        num_atoms=np.int64(num_atoms),
        )

    print(f"Total configurations: {len(data_flat)}")
    print(f"Training configurations: {len(train_indices)}")
    print(f"Validation configurations: {len(val_indices)}")
    print("Train/validation overlap: 0")
    print(f"Global coordinate scale: {global_scale:.6f}")
    print(f"Saved preprocessing: {output_path}")

    return train_normalized, val_normalized

def load_preprocessing(
    preprocessing_file: str,
)-> Dict[str, np.ndarray]:
    """Function loads the preprocessed file, checking if values exist and given them back in a dictonary 

    Args:
        preprocessing_file (str): 

    Returns:
        Dict[str, np.ndarray]: 
    """
    path = Path(preprocessing_file)
    if not path.is_file():
        raise ValueError(f"Preprocessing file not found!")

    with np.load(path) as saved:
        required = ("global_scale", "reference_geometry", "train_indices", "val_indices")
        missing = [name for name in required if name not in saved]
        if missing:
            raise KeyError("Missing preprocessing values:" + ",".join(missing))
        return {name:saved[name].copy() for name in saved.files}

def load_preprocessed_splits(
    data_flat: np.ndarray,
    P: int,
    num_atoms: int,
    preprocessing_file: str,
)-> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load saved preprocessing parameters and apply them to the original data.

    Args:
        data_flat (np.ndarray): _description_
        P (int): _description_
        num_atoms (int): _description_
        preprocessing_file (str): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]: _description_
    """
    parameters = load_preprocessing(preprocessing_file)
    train_indices = parameters["train_indices"].astype(np.int64)
    val_indices = parameters["val_indices"].astype(np.int64)

    if np.intersect1d(train_indices, val_indices).size:
        raise RuntimeError("Saved training and validation splits overlap.")
    if len(train_indices) + len(val_indices) != len(data_flat):
        raise ValueError("Saved split size does not match the current data file.")
    if max(train_indices.max(), val_indices.max()) >= len(data_flat):
        raise IndexError("Saved split indices do not match the current data file.")

    # New preprocessing files include metadata; old compatible files do not.
    if "P" in parameters and int(parameters["P"]) != P:
        raise ValueError("Saved bead count does not match config.P.")
    if "num_atoms" in parameters and int(parameters["num_atoms"]) != num_atoms:
        raise ValueError("Saved atom count does not match config.num_atoms.")

    reference = parameters["reference_geometry"]
    global_scale = float(parameters["global_scale"])
    train_normalized = preprocess_flat(
        data_flat[train_indices], P, num_atoms, reference, global_scale
    )
    val_normalized = preprocess_flat(
        data_flat[val_indices], P, num_atoms, reference, global_scale
    )

    return train_normalized, val_normalized, parameters