import os
import pandas as pd
import numpy as np
from skimage import io, exposure, measure
from skimage.filters import threshold_otsu
from stardist.models import StarDist2D
from stardist.plot import render_label  # Added missing import
from cellpose import models, plot  # Added plot import
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# =====================================================================
# Configuration
# =====================================================================
SAVE_VISUALIZATIONS = True  # True = Visualize and save; False = Skip visualization and do not save

INPUT_FOLDER = 'path/to/your/folder/with/your/merged/RGB/images'
OUTPUT_EXCEL = os.path.join(INPUT_FOLDER, 'cell_counts_results.xlsx')
VISUALIZATION_FOLDER = os.path.join(INPUT_FOLDER, 'visualizations') if SAVE_VISUALIZATIONS else None

# Create visualization folder
if SAVE_VISUALIZATIONS:
    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# =====================================================================
# Initialize Models
# =====================================================================
model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')
model_cellpose = models.Cellpose(gpu=False, model_type='cyto2')

# =====================================================================
# Process All Images
# =====================================================================
results = []

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg')):
        continue

    try:
        # Get filename base
        filename_base = os.path.splitext(filename)[0]  # Added definition

        # Load image
        image_path = os.path.join(INPUT_FOLDER, filename)
        image = io.imread(image_path)

        # Remove alpha channel if present
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Extract channels
        red = image[..., 0]  # Cytoplasm (Red)
        dapi = image[..., 2]  # Nuclei (Blue)

        # =================================================================
        # Process DAPI Channel
        # =================================================================
        dapi_norm = exposure.rescale_intensity(dapi.astype(np.float32), out_range=(0, 1))
        nuclei_labels, _ = model_stardist.predict_instances(dapi_norm, prob_thresh=0.479071, nms_thresh=0.3)
        total_cells = nuclei_labels.max()

        # =============================================================
        # Visualization 1: Nuclei Segmentation
        # =============================================================
        if SAVE_VISUALIZATIONS:
            plt.figure(figsize=(10, 10))
            plt.imshow(render_label(nuclei_labels, img=dapi_norm, alpha=0.7, cmap='jet'))
            plt.title(f'Nuclei Segmentation: {total_cells} cells')
            plt.savefig(os.path.join(VISUALIZATION_FOLDER, f'{filename_base}_nuclei.png'))
            plt.close()

        # =================================================================
        # Process Red Channel
        # =================================================================
        channels = [0, 2]  # [cytoplasm_channel, nucleus_channel]
        cell_masks, _, _, _ = model_cellpose.eval(image, diameter=30, channels=channels)

        # =================================================================
        # Visualization 2: Nuclei-Cytoplasm Association
        # ================================================================
        if SAVE_VISUALIZATIONS:
            plt.figure(figsize=(12, 12))
            plt.imshow(red, cmap='gray')  # Red channel background
            plt.imshow(nuclei_labels > 0, cmap='Greens', alpha=0.3)  # Nuclei
            plt.title(f'Nuclei-Cytoplasm Association: {filename_base}')
            plt.savefig(os.path.join(VISUALIZATION_FOLDER, f'{filename_base}_association.png'))
            plt.close()

        # =================================================================
        # Association and Analysis
        # =================================================================
        cell_props = measure.regionprops(cell_masks)
        cell_centroids = {prop.label: prop.centroid for prop in cell_props if prop.label > 0}

        nuclei_props = measure.regionprops(nuclei_labels)
        nuclei_centroids = [prop.centroid for prop in nuclei_props]

        tree = cKDTree(list(cell_centroids.values()))
        _, cell_indices = tree.query(nuclei_centroids)

        cell_intensity = measure.regionprops_table(cell_masks, intensity_image=red,
                                                   properties=('label', 'mean_intensity'))
        cell_intensity_dict = {lab: intens for lab, intens in zip(cell_intensity['label'],
                                                                  cell_intensity['mean_intensity'])}

        nuclei_intensity = [cell_intensity_dict[list(cell_centroids.keys())[idx]]
                            for idx in cell_indices]

        threshold = threshold_otsu(red)
        positive_cells = np.sum(nuclei_intensity > threshold)
        # =================================================================
        # Positivity Visualization (Add this after threshold calculation)
        # =================================================================
        if SAVE_VISUALIZATIONS:
            # Create figure
            plt.figure(figsize=(12, 6))

            # Subplot 1: Original cytoplasmic staining
            plt.subplot(1, 2, 1)
            plt.imshow(red, cmap='gray')
            plt.title('Original Cytoplasm')

            # Subplot 2: Thresholded positivity
            plt.subplot(1, 2, 2)
            plt.imshow(red > threshold, cmap='gray')
            plt.title(f'Positive Cells (Threshold = {threshold:.1f})')

            # Save and close
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_FOLDER, f'{filename_base}_positivity.png'))
            plt.close()



        # =================================================================
        # Calculate Percentage
        # =================================================================
        percentage = (positive_cells / total_cells * 100) if total_cells > 0 else 0

        # Add to results
        results.append({
            'Filename': filename_base,
            'DAPI+ Cells': total_cells,
            'Cytoplasmic+ Cells': positive_cells,
            'Percentage': round(percentage, 2)
        })

        print(f"Processed {filename}: {total_cells} cells, {positive_cells} positive")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# =====================================================================
# Save Results
# =====================================================================
df = pd.DataFrame(results)
df.to_excel(OUTPUT_EXCEL, index=False)
print(f"Results saved to {OUTPUT_EXCEL}")
