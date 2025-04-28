# Percent-Positive-Cytoplasm
For counting the % positive cells from a merged RGB image in IF stained organoid section.

With organoid sections, you often have cells that overlap each other in immunofluoresence microscopy images. This makes it very difficult for the software that I've tried to count cells, particularly when you are staining for a protein in the cytoplasm. So, this python script uses an already existing Stardist tool which counts nuclei very well from my experience, even with overlaps, and then I added something so it can identify positive cytoplasm based on the nuclei identified.

Added a boolean to save or not save visualization steps (visualization included for your reference, so you can see whether it works with your samples).
