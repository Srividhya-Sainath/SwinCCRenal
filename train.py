import argparse
from pathlib import Path
from base import train
from data import load_train_dataset, load_test_dataset  # Import dataset loading functions

def main(config_path: str):
    """
    Main function to kick off training.
    Args:
        config_path (str): Path to the YAML configuration file.
    """
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

    test_datasets = load_test_dataset(
        geneToAnalyze=gene_to_analyze,
        magLevel=mag_level,
        foldNum=fold_num,
        allSampleDir=all_sample_dir,
        outputPatchDir=outputPatchDir,
        bag_size=None,
    )

    output_path = Path("/mnt/bulk-ganymede/vidhya/crick/SwinApproach")
    output_path.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    learner = train(
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        config_path=config_path,
        path=output_path
    )

    return learner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Swin Transformer model for pathology.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    main(config_path=args.config)