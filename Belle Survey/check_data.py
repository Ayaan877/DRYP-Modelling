from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "Crop_Data_Kharif-Rabi"
CLEAN_DIR = SCRIPT_DIR / "Crop_Data_Cleaned"
SOURCE_FILE = "*_kharif-rabi_crop_data.csv"
CLEAN_FILE = "*_rabi_crop_data_cleaned.csv"

source_files = sorted(SOURCE_DIR.glob(SOURCE_FILE))
cleaned_files = sorted(CLEAN_DIR.glob(CLEAN_FILE))

source = pd.read_csv(source_files[0])
cleaned = pd.read_csv(cleaned_files[0])

print(source.shape)
print(cleaned.shape)