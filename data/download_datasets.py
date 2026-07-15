import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import requests
from io import StringIO
from src.utils.logger import setup_logger

logger = setup_logger("dataset_downloader")


def download_iranian_churn(output_path: Path) -> bool:
    try:
        from ucimlrepo import fetch_ucirepo
        logger.info("Downloading Iranian Churn dataset from UCI...")
        iranian = fetch_ucirepo(id=563)
        df = iranian.data.original
        if df is None:
            df = pd.concat([iranian.data.features, iranian.data.targets], axis=1)
        df.to_csv(output_path, index=False)
        logger.info(f"Iranian Churn saved to {output_path} ({len(df)} rows)")
        return True
    except Exception as e:
        logger.warning(f"ucimlrepo failed: {e}. Trying fallback URL...")
        try:
            url = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                import zipfile, io
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                csv_file = [n for n in z.namelist() if n.endswith(".csv")][0]
                df = pd.read_csv(z.open(csv_file))
                df.to_csv(output_path, index=False)
                logger.info(f"Iranian Churn saved via fallback ({len(df)} rows)")
                return True
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return False


def download_bank_churn(output_path: Path) -> bool:
    urls = [
        "https://raw.githubusercontent.com/erkansirin78/datasets/master/bank_churn/Customer-Churn-Records.csv",
        "https://raw.githubusercontent.com/radheshyamkollipara/bank-customer-churn/main/Customer-Churn-Records.csv",
    ]
    for url in urls:
        try:
            logger.info(f"Trying Bank Churn dataset from {url}...")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                df.to_csv(output_path, index=False)
                logger.info(f"Bank Churn saved to {output_path} ({len(df)} rows)")
                return True
        except Exception as e:
            logger.warning(f"URL {url} failed: {e}")
    logger.error("All bank churn download URLs failed.")
    return False


if __name__ == "__main__":
    import config
    config.IRANIAN_DIR.mkdir(parents=True, exist_ok=True)
    config.BANK_DIR.mkdir(parents=True, exist_ok=True)

    if not config.IRANIAN_FILE.exists():
        download_iranian_churn(config.IRANIAN_FILE)
    else:
        logger.info("Iranian Churn already exists, skipping.")

    if not config.BANK_FILE.exists():
        download_bank_churn(config.BANK_FILE)
    else:
        logger.info("Bank Churn already exists, skipping.")
