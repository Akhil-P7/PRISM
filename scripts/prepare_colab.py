import shutil
from pathlib import Path


def prepare_colab_package():
    print("Preparing PRISM Colab Package...")
    colab_dir = Path("prism-colab")

    if colab_dir.exists():
        shutil.rmtree(colab_dir)

    colab_dir.mkdir()

    # 1. Copy configs
    print("Copying configs...")
    (colab_dir / "configs").mkdir()
    shutil.copy("configs/training.yaml", colab_dir / "configs/training.yaml")

    # 2. Copy models
    print("Copying models...")
    models_dest = colab_dir / "models"
    models_dest.mkdir()

    shutil.copytree("models/shared", models_dest / "shared")
    shutil.copytree("models/cough_detector", models_dest / "cough_detector")
    shutil.copytree("models/disease_classifier", models_dest / "disease_classifier")

    (models_dest / "__init__.py").touch()
    (models_dest / "temporal_transformer").mkdir()
    (models_dest / "temporal_transformer" / "__init__.py").touch()
    (models_dest / "embeddings").mkdir()
    (models_dest / "embeddings" / "__init__.py").touch()

    print(f"Code package created at: {colab_dir.absolute()}")

    # 3. Zip features
    print("\nZipping datasets/features...")
    print(
        "This will take several minutes because the folder is ~6 GB (131,000 files). Please wait..."
    )

    # This creates datasets-features.zip in the current directory
    shutil.make_archive("datasets-features", "zip", "datasets/features")

    print(f"Features zipped to: {Path('datasets-features.zip').absolute()}")
    print("\nALL DONE! You are ready to upload to Google Drive.")


if __name__ == "__main__":
    prepare_colab_package()
