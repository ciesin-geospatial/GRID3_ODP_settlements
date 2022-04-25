# Geospatial Data Analysis Toolset for Settlement Identification

---

***Table of contents***

1. Data Collection
2. Feature Engineering
3. False Positive Classification

---

The toolset presented in this repository aims to predict the possibility that a potential settlement is a true positive (an actual settlement) or a false positive (a non-settlement area that is mistaken as a settlement during the generation stage), in an attempt to improve the accuracy of settlement identification work. 

The generation of candidate settlements is based on image analysis of high-resolution satellite imagery data, which will not be covered in this repo. The toolset here will take in potential settlements and help predict which ones are likely to be false positives. False negatives (actual settlements that are missed in the generation stage) will not be discovered, so it is preferable to be more lenient during the generation process and allow more candidates to be considered in this subsequent filtering stage. It is worth noting that this toolset is developed with small settlements (hamlet or village level) in mind, taking advantage of their relatively smaller scope and simpler geometric shape. However, components of the toolset can be transferred to the filtering of larger settlements. 
