import os,pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import utils as pg

def list_of_bags_collate_fn(batch):
    """
    Custom collate function to keep bags independent.
    Args:
        batch (list): List of tuples (bag, label).
    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]: List of bags and tensor of labels.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    # If the entire batch is invalid, raise an exception to avoid empty training
    if len(batch) == 0:
        return [], torch.tensor([], dtype=torch.long)
    bags, labels = zip(*batch)  # Separate bags and labels
    return list(bags), torch.tensor(labels, dtype=torch.long)

## Taken inspiration from STAMP ##
def _to_fixed_size_bag(bag: torch.Tensor, bag_size: int = 512) -> torch.Tensor:
    """
    Converts a bag of variable size to a fixed size by random sampling and zero-padding.
    Args:
        bag (torch.Tensor): Bag of patches (shape: [num_patches, channels, height, width]).
        bag_size (int): Target fixed bag size.
    Returns:
        torch.Tensor: Fixed-size bag (shape: [bag_size, channels, height, width]).
    """
    num_patches = bag.shape[0]
    if num_patches >= bag_size:
        ## Must I include some form of weighted sampling here? ##
        ## Wherein I include patches that contain more tissue ##
        ## Is that even possible? ##
        bag_idxs = torch.randperm(num_patches)[:bag_size] # Randomly sample bag_size patches
        bag_samples = bag[bag_idxs]
    else:
        # Zero-pad if fewer patches are available
        padding = torch.zeros(bag_size - num_patches, *bag.shape[1:], dtype=bag.dtype)
        bag_samples = torch.cat((bag, padding), dim=0)
    
    return bag_samples


class MILDataset(Dataset):
    def __init__(self, hdf5_list, sample_to_class, pg, bag_size):
        """
        Standard MIL Dataset for pathology with a fixed bag size.
        Args:
            hdf5_list (list): List of HDF5 file paths (one per WSI).
            sample_to_class (np.ndarray): Class labels for each WSI.
            pg: Module to load patches (e.g., `pg.LoadPatchData`).
            bag_size (int): Number of patches to sample per bag.
        """
        self.hdf5_list = hdf5_list
        self.sample_to_class = sample_to_class
        self.pg = pg
        self.bag_size = bag_size

    def __len__(self):
        return len(self.hdf5_list)

    def __getitem__(self, idx):
        hdf5_file = self.hdf5_list[idx]
        class_label = self.sample_to_class[idx]

        try:
            if not os.path.exists(hdf5_file):
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

            patch_data, _, _, _ = self.pg.LoadPatchData([hdf5_file], returnSampleNumbers=True)
            if not patch_data or len(patch_data[0]) == 0:
                raise ValueError(f"No patches found in file {hdf5_file}")
            patches = torch.tensor(patch_data[0], dtype=torch.float32)  # Shape: [num_patches, channels, height, width]

            # If bag_size is None, skip resizing and use all patches
            if self.bag_size is None:
                fixed_bag = patches.permute(0, 3, 1, 2)  # Use all patches directly, permute for PyTorch format
            else:
                # Resize to a fixed-size bag using _to_fixed_size_bag
                fixed_bag = _to_fixed_size_bag(patches.view(patches.shape[0], -1), self.bag_size)
                fixed_bag = fixed_bag.view(self.bag_size, 224, 224, 3).permute(0, 3, 1, 2)

            class_tensor = torch.tensor(class_label, dtype=torch.long)

            return fixed_bag, class_tensor

        except (FileNotFoundError, ValueError) as e:
            # Log the error and skip this file
            with open("faulty_files.log", "a") as log_file:
                log_file.write(f"Skipping file {hdf5_file}: {str(e)}\n")
            return None


def load_train_dataset(geneToAnalyze, magLevel, foldNum, allSampleDir, outputPatchDir, bag_size):
    """
    Efficient PyTorch-based data loader for patches.
    Args:
        geneToAnalyze (str): Gene to analyze (e.g., BAP1).
        magLevel (str): Magnification level (should be '20X').
        foldNum (int): Fold number for cross-validation (0, 1, or 2).
        allSampleDir (str): Directory containing CSV files for folds.
        outputPatchDir (str): Directory with non-focal patches.
        batch_size (int): Number of patches to load at a time.
    Returns:
        dict: Training and testing data dictionaries.
    """
    if magLevel != '20X':
        raise ValueError("Magnification should be 20X")

    # Map fold numbers to file names
    fold_train_file = os.path.join(allSampleDir, f"fold{foldNum}_train.csv")

    trainSamples = pd.read_csv(fold_train_file)

    trainSamples = trainSamples[trainSamples['Number_Patches_Extracted'] != 0]

    expected_rows = {0: 513, 1: 515, 2: 522}
    if len(trainSamples) != expected_rows[foldNum]:
        raise ValueError(f"Fold {foldNum}: Expected {expected_rows[foldNum]} rows, but found {len(trainSamples)} in the training set.")

    def get_hdf5_files(samples):
        hdf5_list = []
        sample_to_class = []

        for _, row in samples.iterrows():
            hdf5_file = os.path.join(outputPatchDir, row['svs'].replace('.svs', '.hdf5'))
            if not os.path.exists(hdf5_file):
                raise FileNotFoundError(f"Missing HDF5 file: {hdf5_file}")
            
            hdf5_list.append(hdf5_file)
            sample_to_class.append(row[geneToAnalyze + '_Positive'])

        return hdf5_list, np.uint8(sample_to_class)

    trainHdf5List, trainSampleToClass = get_hdf5_files(trainSamples)

    trainDataset = MILDataset(trainHdf5List, trainSampleToClass, pg, bag_size=bag_size)

    return {"train": trainDataset}

def load_test_dataset(geneToAnalyze, magLevel, foldNum, allSampleDir, outputPatchDir, bag_size):
    """
    Efficient PyTorch-based data loader for patches.
    Args:
        geneToAnalyze (str): Gene to analyze (e.g., BAP1).
        magLevel (str): Magnification level (should be '20X').
        foldNum (int): Fold number for cross-validation (0, 1, or 2).
        allSampleDir (str): Directory containing CSV files for folds.
        outputPatchDir (str): Directory with non-focal patches.
        batch_size (int): Number of patches to load at a time.
    Returns:
        dict: Testing data dictionaries.
    """
    if magLevel != '20X':
        raise ValueError("Magnification should be 20X")

    fold_test_file = os.path.join(allSampleDir, f"fold{foldNum}_test.csv")

    testSamples = pd.read_csv(fold_test_file)

    testSamples = testSamples[testSamples['Number_Patches_Extracted'] != 0]

    expected_rows = {0: 262, 1: 260, 2: 253}
    if len(testSamples) != expected_rows[foldNum]:
        raise ValueError(f"Fold {foldNum}: Expected {expected_rows[foldNum]} rows, but found {len(testSamples)} in the training set.")

    def get_hdf5_files(samples):
        hdf5_list = []
        sample_to_class = []

        for _, row in samples.iterrows():
            hdf5_file = os.path.join(outputPatchDir, row['svs'].replace('.svs', '.hdf5'))
            if not os.path.exists(hdf5_file):
                raise FileNotFoundError(f"Missing HDF5 file: {hdf5_file}")
            
            hdf5_list.append(hdf5_file)
            sample_to_class.append(row[geneToAnalyze + '_Positive'])

        return hdf5_list, np.uint8(sample_to_class)

    testHdf5List, testSampleToClass = get_hdf5_files(testSamples)

    testDataset = MILDataset(testHdf5List, testSampleToClass, pg, bag_size=bag_size)

    return {"test": testDataset}

# def load_train_dataset(geneToAnalyze, magLevel, foldNum, foldsIdx, allSampleFile, outputPatchDir, focalPatchDir, bag_size):
#     """
#     Efficient PyTorch-based data loader for patches.
#     Args:
#         geneToAnalyze (str): Gene to analyze (e.g., BAP1).
#         magLevel (str): Magnification level (should be '20X').
#         foldNum (int): Fold number for cross-validation.
#         foldsIdx (str): Path to pickle file with fold indices.
#         allSampleFile (str): Path to CSV file with sample metadata.
#         outputPatchDir (str): Directory with non-focal patches.
#         focalPatchDir (str): Directory with focal patches.
#         batch_size (int): Number of patches to load at a time.
#     Returns:
#         dict: Training and testing data dictionaries.
#     """
#     if magLevel != '20X':
#         raise ValueError("Magnification should be 20X")
    
#     # Load fold indices
#     with open(foldsIdx, 'rb') as f:
#         folds = pickle.load(f)
    
#     testIdx = folds[foldNum]  # Test indices
    
#     # Determine training indices
#     if foldNum == 0:
#         trainIdx = np.array(list(folds[1]) + list(folds[2]))
#     elif foldNum == 1:
#         trainIdx = np.array(list(folds[0]) + list(folds[2]))
#     elif foldNum == 2:
#         trainIdx = np.array(list(folds[0]) + list(folds[1]))
    
#     # Load sample data
#     allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
#     # Added by me
#     allSamples = allSamples[allSamples['Number_Patches_Extracted'] != 0]  # Discard rows with 0 in Number_Patches_Extracted

#     # Added by me
#     available_trainIdx = [idx for idx in trainIdx if idx < len(allSamples)]
#     missing_trainIdx = [idx for idx in trainIdx if idx >= len(allSamples)]

#     if missing_trainIdx:
#         print(f"Warning: The following train indices are missing in the dataset and will be skipped: {missing_trainIdx}")

#     # Extract training nonFocal samples
#     trainSamples = allSamples.iloc[available_trainIdx]
#     trainNonFocalSamples = trainSamples.iloc[np.where(trainSamples[geneToAnalyze + '_Focal'].values == False)[0]]
#     trainHdf5ListNF = [os.path.join(outputPatchDir, f.replace('.svs', '.hdf5')) for f in trainNonFocalSamples.svs.values]
#     trainSampleToClassNF = np.uint8(trainNonFocalSamples[geneToAnalyze + '_Positive'].values)
    
#     # Handle "Focal" samples for BAP1
#     if geneToAnalyze == 'BAP1':
#         trainFocalSamples = trainSamples.iloc[np.where(trainSamples[geneToAnalyze + '_Focal'].values == True)[0]]
#         trainHdf5ListF = [os.path.join(focalPatchDir, f.replace('.svs', '.hdf5')) for f in trainFocalSamples.svs.values]
#         trainSampleToClassF = np.uint8(trainFocalSamples[geneToAnalyze + "_Positive"].values)
#         trainDataset = MILDataset(trainHdf5ListNF + trainHdf5ListF, 
#                                         np.concatenate((trainSampleToClassNF, trainSampleToClassF)), 
#                                         pg, bag_size)
#     else:
#         print(trainHdf5ListNF, trainSampleToClassNF)
#         trainDataset = MILDataset(trainHdf5ListNF, trainSampleToClassNF, pg, bag_size)
    
    # return {"train": trainDataset}
    

def main():
    from torch.utils.data import DataLoader
    gene_to_analyze = "PBRM1"
    mag_level = "20X"
    fold_num = 0
    all_sample_dir = "/mnt/bulk-ganymede/vidhya/crick/models/acosta/Folds"
    outputPatchDir = '/mnt/bulk-ganymede/vidhya/crick/docker/Patch_Data/WSI/'
    bag_size = 128

    train_datasets = load_train_dataset(
        geneToAnalyze=gene_to_analyze,
        magLevel=mag_level,
        foldNum=fold_num,
        allSampleDir=all_sample_dir,
        outputPatchDir=outputPatchDir,
        bag_size=bag_size,
    )

    # test_datasets = load_test_dataset(
    #     geneToAnalyze=gene_to_analyze,
    #     magLevel=mag_level,
    #     foldNum=fold_num,
    #     allSampleDir=all_sample_dir,
    #     outputPatchDir=outputPatchDir,
    #     bag_size=None,
    # )
    train_loader = DataLoader(train_datasets["train"], batch_size=12, collate_fn=list_of_bags_collate_fn,shuffle=True)
    #test_loader = DataLoader(test_datasets["test"], batch_size=1, shuffle=False, collate_fn=list_of_bags_collate_fn)

    for i, (patches, labels) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"- Number of samples: {len(patches)}")
        for j, patch in enumerate(patches):
            print(f"  Sample {j+1} patch shape: {patch.shape}")
        print(f"- Labels: {labels}")

if __name__ == "__main__":
    main()