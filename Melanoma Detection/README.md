# Melanoma_Detection_Case_Study

> In this case study, we have build a multiclass classification model using a custom convolutional neural network in TensorFlow. 


## Table of Contents
* [Problem statement](#problem-statement)
* [Acknowledgements](#project-pipeline)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)


<!-- You can include any other section that is pertinent to your problem -->

## Problem statement
 To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

 The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Project Pipeline
- Data Reading/Data Understanding
- Dataset Creation
- Dataset visualisation
- Model Building & training 
- Chose an appropriate data augmentation strategy to resolve underfitting/overfitting
- Model Building & training on the augmented data
- Class distribution
- Handling class imbalances
- Model Building & training on the rectified class imbalance data


## Conclusions
- The problem of overfitting and underfitting was solved and the model was well trained for predictions. Data augmentation, outliers, and class equalization were found to be useful in improving model performance in this case.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Keras
- TensorFlow
- Python 3
- Pandas, Numpy, Matplotlib,
- Augmentor

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
