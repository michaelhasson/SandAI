# Automated determination of transport and depositional environments in sand and sandstones

## Description
This repository includes all code related to the manuscript "Automated determination of transport and depositional environments in sand and sandstones" by Michael Hasson, M. Colin Marvin, and Mathieu Lapôtre. Using the files in this repository, anyone can use the model documented in our manuscript to identify the transport or depositional environment of scanning electron microscope (SEM) images of individual grains of quartz sand (as long as the meet the requirements for model input). The only configuration required is for a user to set the path to their folder of images and path where they would like the results to be saved.

The model has been validated on modern and ancient quartz grains, so it can be used on modern sediment and lithified rocks. 

Included are: 
- Tutorials for use with example outputs
- All code related to training and evaluating the model used in this study

## Checklist

Before using the model, it is critical to make that images are suitable for model inference. Failure to do so will lead to inaccurate results. The requirements are:
1. Sand must be from terrestrial environments (eolian, glacial, beach, or fluvial).
2. Individual grains only -- there can only be one sand grain per image.
3. No scale bars.
4. The original grain shapes and textures must be present. If they have been obscured by diagenesis (e.g., silica cementation) or the sample preparation procedure, they will not produce valid results.

### Visual checklist examples:
Make sure that your images *do not* look like these before using the classifier!

Individual grains only? 
![plot](./Checklist_images/Scale_bars/scale_bar.png)
