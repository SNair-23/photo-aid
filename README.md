# Photo-aid
A deep-learning based application that recommends changes to portraits based off of the factors of blur, occlusion, make up, illumination, expression, and pose. Ideal for headshot photography. 

# Prerequisites
- Linux OS
- Python 3
- Pandas Library

# Model Training
Complete information on the WFLW dataset can be found at https://wywu.github.io/projects/LAB/WFLW.html 


LOADING ANNOTATIONS AND IMAGES:
1. Download the WFLW Face Annotations
~~~
wget -P ./Downloads/ "https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz"
~~~
2. Unzip them into 'Downloads/WFLW_files/datasets/'
~~~
mkdir ./Downloads/WFLW_files/datasets/WFLW
mv ./Downloads/WFLW_annotations.tar.gz ./Downloads/WFLW_files/datasets/WFLW
cd ./Downloads/WFLW_files/datasets/WFLW
tar -xvzf WFLW_annotations.tar.gz
~~~
3.Download the annotations_parser.py file from above and run it with:
~~~
python3 ~./Downloads/annotations_parser.py
~~~
Enter "R" to obtain the WFLW training data

5. Download WFLW Images from https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view

*The script used for training can be found as wflw_nn_trainer_colab.py in this repo. Keep in mind that filepaths and package installations may have to be manipulated as this script was written in Google Colab for computational speed purposes.*

IF USING GOOGLE COLAB:
- Training code at https://colab.research.google.com/drive/1r_dXCJkMsFt6vrwoZXCks5hQXr6LpyDq?usp=sharing --- Create a copy to edit
- Upload the saving_trainer.csv file into your Colab Runtime files and allow for your Google Drive to be mounted when running the training.
- Ensure that your WFLW_images.tar.gz file from the image download is uploaded to your Google Drive
- The model for the executed image attribute will be outputed to a .keras file within your run time after all epochs are complete.
- Download the .keras file to save your model when satisfied with training results

TRIALS:
- *Manipulate Epochs, Learning Rate, and Batch Size to view accuracies of different attributes*
- Function: run_5_times(csv_data, epoch, attribute) --- Compare the speed of program in 5 runs with 20 epochs VS 1 run of 100 epochs
- Function: run_all_fwd_rev(csv_data, epoch) --- Ensures that order of attributes listed does not effect rate of loss

# Model Testing
1. Model Can be tested on WFLW Testing data. Again run:
   ~~~
   python3 ~./Downloads/annotations_parser.py
   ~~~
   (Type "E" to load testing annotations)

   - Upload the csv derived from annotations_parser.py as the input csv file to evaluate the model of each attribute.
   - Ensure that the desired .keras models are uploaded to Google Drive, altering the filepath as needed.
   - 

2. You can also test the accuracy of the model for illumination using your own image files.
   - Download the "image_lighting_calc.py" file and assign the absolute path to your Google Drive folder of images to "image_folder".
   - Download the csv created from the impage-lighting-calc.py script and pass it into the WFLW_NN_Tester_colab.py file

# Additional Files:
> image_lighting_calc.py --- Script to find lighting values of input images and store them in a csv file
> Lighting-Trainer.ipynb --- Uses csv file of lightings developed from previous file to train a model to recognize new images' lighting values on a scale from 0-2
