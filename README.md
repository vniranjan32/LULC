
import rasterio
import numpy as np
import joblib

RGB_PATH   = r"E:\CNN_Input\Clippedfile.tif"
MODEL_PATH = r"E:\IMG\outputs_change\rf1_lulc_model.pkl"
OUTPUT_TIF = r"E:\IMG\outputs_change\lulc1prediction.tif"


rf_model = joblib.load(MODEL_PATH)
print("✅ RF model loaded")
print("Model classes:", rf_model.classes_)

src = rasterio.open(RGB_PATH)

profile = src.profile
profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw",
    nodata=0
)

with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:

    total_blocks = sum(1 for _ in src.block_windows(1))
    processed = 0

    for ji, window in src.block_windows(1):

        rgb_block = src.read(window=window)

        bands, h, w = rgb_block.shape
        X_block = rgb_block.reshape(bands, -1).T

        if X_block.size == 0:
            continue

        pred_block = rf_model.predict(X_block)

        pred_block = pred_block.reshape(h, w).astype(np.uint8)

        dst.write(pred_block, 1, window=window)

        processed += 1
        if processed % 200 == 0:
            print(f"Processed {processed}/{total_blocks} blocks")

src.close()

print("🎉 LULC classification completed successfully")
print("Output saved at:", OUTPUT_TIF)
