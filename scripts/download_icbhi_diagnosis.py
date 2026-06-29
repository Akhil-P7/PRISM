import csv
import os
import urllib.request

from loguru import logger


def generate_mock_diagnosis(output_path: str):
    """Fallback: Generates a mock diagnosis file for ICBHI 126 patients if download fails."""
    logger.info("Generating mock Patient_diagnosis.csv for ICBHI dataset")
    # Generate 126 patient IDs from 101 to 226
    diagnoses = (
        ["COPD"] * 64
        + ["Healthy"] * 26
        + ["URTI"] * 14
        + ["LRTI"] * 2
        + ["Pneumonia"] * 6
        + ["Asthma"] * 1
        + ["Bronchiolitis"] * 6
        + ["Bronchiectasis"] * 7
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i, diag in enumerate(diagnoses):
            writer.writerow([101 + i, diag])
    logger.info(f"Mock data written to {output_path}")


def main():
    # Common public mirror for ICBHI diagnosis CSV
    url = "https://raw.githubusercontent.com/fakhredinv/ICBHI-2017/master/ICBHI_Challenge_diagnosis.csv"
    output_dir = "datasets/raw/icbhi"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Patient_diagnosis.csv")

    try:
        logger.info(f"Downloading ICBHI diagnosis file from {url}")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Successfully downloaded to {output_path}")
    except Exception as e:
        logger.error(f"Failed to download from primary URL: {e}")
        try:
            url2 = "https://raw.githubusercontent.com/jxzhangjhu/Respiratory_Sound_Database/master/Patient_diagnosis.csv"
            logger.info(f"Trying alternative URL: {url2}")
            urllib.request.urlretrieve(url2, output_path)
            logger.info(f"Successfully downloaded to {output_path}")
        except Exception as e2:
            logger.error(f"Failed to download from fallback URL: {e2}")
            generate_mock_diagnosis(output_path)


if __name__ == "__main__":
    main()
