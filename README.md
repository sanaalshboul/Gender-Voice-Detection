
Gender detection has been trained on Voice-gender dataset [(download)](https://www.kaggle.com/liftofff9/voice-gender "download") , which consists of 10 training audio files, 5 of them are female and the rest for males. In addition to, a testing set of 558 females and 564 males. The duration of every audio file in the training set is 1 minute, so the whole training duration is 10 minutes. The duration of every file in the testing set is 10 seconds, which gives about 3 hours and 8 minutes for the testing set audio duration. 

Gender/set    | Training set  | Tesing set
------------- | ------------- | -------------
Num of female |       5       |     558
Num of male   |       5       |     546

Time/set      | Training set  | Tesing set
------------- | ------------- | ---------------------
Time duration |   10 minutes  | 3 hours and 8 minutes

The main language that was used for recording the audio in the dataset is English. The number of utterances in the dataset is 1114 utterances. However, the utterance length is different from one audio to another.

## Requirments
* Pandas
* Numpy
* Matplotlib
* Sklearn
* Scipy
* Librosa
* Keras
* Tensorflow
* Xgboost

## Explanatory analysis and audio visualization
Running [ELA_Voice_visualization.ipynb ](https://colab.research.google.com/drive/1sJG-8M3Fj16OYZtvkcqL-9XUIDPmkXP6?authuser=4 "download") shows some information about the number of utterances and the whole time duration about the dataset. Moreover, it shows voice and spectrum visualization for male and female. 

The left figures show that the signal magnitude of the female audio is larger than the magnitude for male audio. However, The right figures show the spectrogram of female and male audio. The spectrogram represents the wave in time-frequency domain, which produces the intensities of accuring freuencies over the time. The audio male frequency is larger than the frequency of the female audio as shown in the spectrum figures (since the lighter region in the spectrogram gives a higher frequencies).

![picture alt](https://github.com/sanaalshboul/Gender-Voice-Detection/blob/master/images/spectrum.png "Title is optional")

This left figures show the spectrum magnitude of the male and female audio, which produces the audio energy. The energy or the spectrum magnitude of male audio is larger than the spectrum magnitude of the female audio.

![picture alt](https://github.com/sanaalshboul/Gender-Voice-Detection/blob/master/images/magnitude_spec.png "Title is optional")

## Feature Extraction and data preperation
Running [Data_preperation.ipynb ](https://colab.research.google.com/drive/1PPHHSDRQd_74A32sByNWE1zJ98ejVmK0?authuser=4 "download") prepare the extracted features and extracts some audio features such as mfcc, mel, tonnetz, chroma, and contrast. In this work mfcc feature is extracted from the training and the testing sets and the features of all audio files are saved as ([all_mfcc_train.npy](https://github.com/sanaalshboul/Gender-Voice-Detection/blob/master/data/all_mfcc_train.npy "download") for the training set and [all_mfcc_test.npy](https://github.com/sanaalshboul/Gender-Voice-Detection/blob/master/data/all_mfcc_test.npy "download") for the testing set), while the gender type of each audio is saved in [all_label_train.npy](https://github.com/sanaalshboul/Gender-Voice-Detection/blob/master/data/all_label_train.npy "download") and [all_label_test.npy](https://github.com/sanaalshboul/Gender-Voice-Detection/blob/master/data/all_label_test.npy "download"). In addition, the feature for every audio file are saved in the [data/mfcc_train](https://github.com/sanaalshboul/Gender-Voice-Detection/tree/master/data/mfcc_train "download") and [data/mfcc_test](https://github.com/sanaalshboul/Gender-Voice-Detection/tree/master/data/mfcc_test "download") directories, so you don't need to extract mfcc features again. To change and save another feature files change the arguments of these methods:

* for training:
`data_feature_preperation(train_or_test="train", feature_name="mel")`,   `save_data(train_or_test="train")`
* for testing:
`data_feature_preperation(train_or_test="test", feature_name="mel")`,   `save_data(train_or_test="test")`

## Training and testing
Running [Data_preperation.ipynb ](https://colab.research.google.com/drive/1MSUFcVmZV0R7BBJGXEcDvYcMKNus6kbG?authuser=4 "download") train and test the dataset on several classifiers and a Neural Network(NN). The hyperparameters of XGBclassifier, DecisionTreeClassifier and the NN were choosen using Gridsearch for hyperparameters tuning. The test result of each classifier are shown in the following tabel: 

Algorithm                    | Accuracy score  |  Precision  | Recall  |
---------------------------  | --------------- | ----------- | ------- |
XGBClassifier                |       68%       |     69      |   68    |
GradientBoostingClassifier   |       63%       |     65      |   62    |
DecisionTreeClassifier       |       57%       |     57      |   57    |
NN                           |       59%       |     --      |   --    |

The XGBClassifier achieved the higher accuracy score (68%). This means that 68% of the time the classifier is able to detect if the voice is for male or female.

Precision measures the propotion of accurate positive predictions out of all positive prediction made , so the higher precision classifier is better. XGBClassifier has the higher precision. Recall measures the propotion of accurate positive predictions out of all actual positive observations, and XGBClassifier achieved the higher recall.

