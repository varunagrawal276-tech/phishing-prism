import base64
from pathlib import Path

# Read base64
with open("kaggle_deploy/dataset/src.zip", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

# Read existing train_on_gpu.py
with open("kaggle_deploy/kernel/train_on_gpu.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace src_zip extraction logic with embedded base64 extraction
old_extract = """# Unzip source code if present
src_zip = DATA_DIR / "src.zip"
if src_zip.exists():
    logger.info("Extracting src.zip to /kaggle/working/...")
    with zipfile.ZipFile(src_zip, "r") as z:
        z.extractall(WORKING_DIR)"""

new_extract = f"""# Embedded source code
import io
import base64
import zipfile
SRC_B64 = "{b64}"
logger.info("Extracting embedded src package...")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SRC_B64))) as z:
    z.extractall(WORKING_DIR)"""

if old_extract in code:
    code = code.replace(old_extract, new_extract)
else:
    print("Warning: old_extract not found exactly, doing import replacement")
    code = code.replace(
        'from src.features.feature_pipeline import FeaturePipeline',
        f'# Embedded source\nimport io\nimport base64\nimport zipfile\nSRC_B64 = "{b64}"\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode(SRC_B64))) as z:\n    z.extractall(WORKING_DIR)\nsys.path.insert(0, str(WORKING_DIR))\nfrom src.features.feature_pipeline import FeaturePipeline'
    )

# Dynamic parquet finder
old_parquets = """train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df = pd.read_parquet(DATA_DIR / "val.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")"""

new_parquets = """all_parquets = list(Path("/kaggle/input").rglob("*.parquet"))
logger.info(f"Discovered input parquets: {[p.name for p in all_parquets]}")
train_p = next(p for p in all_parquets if "train" in p.name)
val_p = next(p for p in all_parquets if "val" in p.name)
test_p = next(p for p in all_parquets if "test" in p.name)
train_df = pd.read_parquet(train_p)
val_df = pd.read_parquet(val_p)
test_df = pd.read_parquet(test_p)"""

if old_parquets in code:
    code = code.replace(old_parquets, new_parquets)

with open("kaggle_deploy/kernel/train_on_gpu.py", "w", encoding="utf-8") as f:
    f.write(code)

print("kaggle_deploy/kernel/train_on_gpu.py successfully embedded with src and dynamic path resolution!")
