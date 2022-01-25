# DEEP LEARNING PROJECT WRITE-UP

## CHEST X-RAY CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS 

## Abstract

Tinytown is a rural hamlet in northern Pennsylvania with a population of approximately 5000. Having no formal hospital, the local residents rely on Tinytown Health Center, a small medical office staffed by just one general practitioner (M.D.) and a registered nurse. While the staff of Tinytown Health does have the equipment to take chest X-rays, they lack the expertise to confidently diagnose some of the severe conditions which such X-rays can reveal (especially conditions like Atelectasis, Cardiomegaly, Edema, and Pleural Effusion, which have inordinately affected the town's elderly-skewing population in the past). 

Concerned about the potential for missing such diagnoses, Tinytown Health would like to commission a Deep Learning-based classification system which can flag locally-administered chest X-rays, alerting the staff when such images may indicate serious conditions that justify immediate electronic transmission of the X-ray(s) to an on-call radiology expert in a Scranton area hospital (100 miles away) who can make the final, potentially life-saving, diagnostic determination.

## Design

Tinytown health will be able to feed a digitized 224 x 273 pixel greyscale representation of each chest X-ray they take into the system and get back four classification determinations, each "Negative" or "Possible Positive" relative to one of the four conditions of interest (Atelectasis, Cardiomegaly, Edema, and Pleural Effusion). These determinations will be made using Convolutional Neural Network (CNN) classifier models built in Keras and Tensorflow and trained/validated/tested on a multi-diagnsotic-labeled dataset compiled by Stanford University School of Medicine and made available through their Machine Learning Group via this site: https://stanfordmlgroup.github.io/competitions/chexpert/. 

## Data

The aforementioned dataset consists of 224,316 chest X-rays of 65,240 patients, performed between October 2002 and July 2017 in both inpatient and outpatient centers. Each X-ray is labeled for the diagnosis of multiple possible conditions based on the contents of its associated radiology report. The images are provided as 320 x 390 grayscale JPEGs. These have been resampled to dimensions of 224 x 273 to enable more time-efficient training of the Deep Learning classification models.

For the purposes of this project, the downloaded dataset was filtered down to just frontally-shot chest X-rays (views from the back and side are present in the full dataset), limited to just one X-ray per patient, and only diagnostic labels for Atelectasis, Cardiomegaly, Edema, and Pleural Effusion were retained, resulting in 53,528 images with 4 diagnostic labels each.

- An individual unit of data has the following characteristics:
  - Image Representation (tensor)  :  a tensor representation of an X-ray image having shape (224, 273, 1)
  - Target_0 (binary)                        :  0 = Negative for Atelectasis, 1 = Possible Positive for Atelectasis
  - Target_1 (binary)                        :  0 = Negative for Cardiomegaly, 1 = Possible Positive for Cardiomegaly
  - Target_2 (binary)                        :  0 = Negative for Edema, 1 = Possible Positive for Edema
  - Target_3 (binary)                        :  0 = Negative for Pleural Effusion, 1 = Possible Positive for Pleural Effusion

## Algorithms

Baseline models using traditional Machine Learning classification algorithms (Random Forest and Logistic Regression) were initially built as points of comparison for the CNN models. The baseline models all demonstrated weak AUROCs (0.636 best case for one target and less than 0.60 for the other three).

A 4-label CNN classifier was built and did considerably better than the baselines, but for every target its performance was beaten by using four indifidual CNN models (one per target), each hyperparameter (dropout, l1 and l2 regularization) tuned for a binary classification task. All CNN models made use of Transfer Learning based on a VGG-16 model pre-trained on the ImageNet dataset. The best-performing CNNs allowed the final block (block 5) of the VGG-16 to train its convolutional layer weights on the X-ray dataset to achieve slighly better performance than what was seen with all VGG-16 layers frozen.

## Tools 

The following tools were used in this project:

1. Pandas to clean, explore and generate the final modeling data
2. SKLearn to build traditional Machine Learning classification models as well as to perform data splitting for model development and  to measure model performance 
3. Keras and TensorFlow 2 to build Deep learning classification models
4. Matplotlib and Seaborn to generate any needed visualizations
5. Python 3.8 (to be able to use all of the above)

## Communication

In addtition to presenting final Project Slides to the stakeholders, all work (including the slides) can be found on GitHub: https://github.com/georgepappy/DeepLearning

