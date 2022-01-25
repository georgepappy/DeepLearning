# PROJECT PROPOSAL

##### A DEEP LEARNING CLASSIFIER FOR CHEST X-RAYS

Tinytown is a rural hamlet in northern Pennsylvania with a population of approximately 5000. Having no formal hospital, the local residents rely on Tinytown Health Center, a small medical office staffed by just one general practitioner (M.D.) and a registered nurse. While the staff of Tinytown Health does have the equipment to take chest X-rays, they lack the expertise to confidently diagnose some of the severe conditions which such X-rays can reveal (especially conditions like Atelectasis, Cardiomegaly, Edema, and Pleural Effusion, which have inordinately affected the town's elderly-skewing population in the past). 

Concerned about the potential of missing such diagnoses, Tinytown Health would like to commission a Deep Learning-based classification system which can flag locally-administered chest X-rays, alerting the staff when such images may indicate serious conditions that justify immediate electronic transmission of the X-ray(s) to an on-call radiology expert in a Scranton area hospital (100 miles away) who can make the final, potentially life-saving, diagnostic determination.

The dataset for this project was compiled by Stanford University School of Medicine and has been downloaded from their Machine Learning Group via this site: https://stanfordmlgroup.github.io/competitions/chexpert/. It consists of 224,316 chest X-rays of 65,240 patients, performed between October 2002 and July 2017 in both inpatient and outpatient centers. Each X-ray is labeled for the diagnosis of multiple possible conditions based on the contents of its associated radiology report. 

For the purposes of this project, the downloaded dataset has been filtered down to just frontally-shot chest X-rays (views from the back and side views are present in the full dataset), limited to just one X-ray per patient, and only diagnostic labels for Atelectasis, Cardiomegaly, Edema, and Pleural Effusion have been retained, resulting in 53,528 images with 4 diagnostic labels each. These labels have been translated into 16 (=2^4) unique and mutually exclusive targets. For instance, all four labels equal to 0 (negative diagnosis for all 4 conditions) maps to target 0, whereas all four labels equal to 1 (positive diagnosis for all 4 conditions) maps to target 15.

The images are provided as 320 x 390 grayscale JPEG images. These have been resampled to a dimension of 224 x 273 to enable more time-efficient training of the various Deep Learning classification models which will be built.

An individual unit of data has the following characteristics:

- Image Representation (tensor) : a tensor representation of an X-ray image having shape (224, 273, 1)

- Target (integer) : An integer ranging from 0-15 representing anywhere from zero to four simultaneous diagnoses;

  ​                          Target = *Atelectasis* x 2^3 + *Cardiomegaly* x 2^2 + *Edema* x 2^1 + *Pleural Effusion*

  ​								where each of the diagnosable conditions takes values 1='positive diagnosis' or  0='negative diagnosis'		

As previously indicated, this project will use the dataset described above to generate a variety of Deep Learning Convolutional Neural Network classifiers to identify patient X-rays which may indicate serious conditions which call for further image analysis by a radiological expert. In addition to trying various CNN model topologies and other hyperparameter tuning experiements (including optimizer choice and settings), the project will attempt to make use of transfer learning if appropriate pre-trained weights and an associated model can be found. A baseline Logistic Regression model will also be built to provide a performance point of reference. 

The tools required for this project are: 

1. Pandas to clean, explore and generate the final modeling data
2. SKLearn, TensorFlow and Keras to implement various models and handle/transform the image data as suitable tensors
3. Matplotlib and Seaborn to generate any needed visualizations
4. Python 3.8 (to be able to use all of the above)

The Minimum Viable Product for this project will be a report (and slides) presenting a final Deep Learning model with meaningful performance metrics such as Accuracy and AUC (overall average and per specific diagnosis).

