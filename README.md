# rooftopPV_dbsmR2025
The EU Digital Building Stock Model (DBSM) is designed to provide a comprehensive and detailed geospatial database of individual buildings within the European Union, focusing primarily on their energy-related characteristics, although DBSM has a wide range of applications. The main goal of this initiative is to support key energy policies, such as the Energy Performance of Buildings Directive, by facilitating more targeted and informed investment decisions in the context of the building renovation wave. DBSM also aims to support the recent European Affordable Housing initiative, for example by enabling more precise "what-if" analyses and assessments of higher granularity. 


The repositoty has 4 scripts (step0, step1, step2, step3) that the user can use and replicate the conflation of DBSM R2025 (step 0) and the estimation of the rooftop PV potential (step3). 
1)  Step0 this is the core code of DBSM R2025. To run this the user needs as input the EUBUCCO, OpenSteetMap and Microsoft Buildings .gpkg or .fgb (vector format) per country
2) Step1 is the postprocessing after the conflation and contains three main steps: 
  a) the identification of duplicates and overapping buildings,
  b) identification of large porisitve overlaid with water or forest parcels
  c) identification of false positive and small buildingd <5m2 area.
3) Step 2, inlcudes the assignment of the unique code id and final check for duplicates, cleaning and homoginize the attribute table, basic statics.
4) Step 3, estimates the rooftop PV potential at building level and then exported as a .gpkg at country level.
More detailed description can be found in https://op.europa.eu/en/publication-detail/-/publication/d418ba32-473e-11f0-85ba-01aa75ed71a1/language-en
The detailed description of the attribute fields are presetned in the ANNEX of paper: LINK HERE.
