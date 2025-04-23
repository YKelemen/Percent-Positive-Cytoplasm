import os
import pandas as pd
import numpy as np
import openpyxl
from skimage import io, exposure, measure
from skimage.filters import threshold_otsu
from stardist.models import StarDist2D
from cellpose import models
from scipy.spatial import cKDTree

# =====================================================================
# Configuration
# =====================================================================
INPUT_FOLDER = 'C:/Users/Cardboard/Desktop/Work/Papers/Manuscript/ACE Proposal/Figures/GINS images'
OUTPUT_EXCEL = os.path.join(INPUT_FOLDER, 'cell_counts_results.xlsx')

# =====================================================================
# Initialize Models (load once for efficiency)
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
        # Load image
        image_path = os.path.join(INPUT_FOLDER, filename)
        image = io.imread(image_path)

        # Remove alpha channel if present
        if image.shape[-1] == 4:
            image = image[..., :3]

        # =======================================
        # Change red to green by changing 0 to 1
        # =======================================
        # Extract channels
        red = image[..., 0]  # Cytoplasm (Red)
        dapi = image[..., 2]  # Nuclei (Blue)

        # =================================================================
        # Process DAPI Channel
        # =================================================================
        dapi_norm = exposure.rescale_intensity(dapi.astype(np.float32), out_range=(0, 1))
        nuclei_labels, _ = model_stardist.predict_instances(dapi_norm, prob_thresh=0.479071, nms_thresh=0.3)
        total_cells = nuclei_labels.max()

        # =================================================================
        # Process Red Channel
        # =================================================================
        channels = [0, 2]  # [cytoplasm_channel, nucleus_channel]
        cell_masks, _, _, _ = model_cellpose.eval(image, diameter=30, channels=channels)

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
        # Calculate Percentage
        # =================================================================
        percentage = (positive_cells / total_cells * 100) if total_cells > 0 else 0

        # Add to results
        results.append({
            'Filename': os.path.splitext(filename)[0],
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
