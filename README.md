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
tar -xvzf WFLW_annotations.tar.gz | ./Downloads/WFLW_Files/datasets/WFLW
~~~
3. Download and run annotations_parser.py, and type "R" for training. 
~~~
wget -P ./Downloads/ "https://wywu.github.io/projects/LAB/support/annotations_parser.py"
python3 annotations_parser.py
~~~
Training data should now be stored in a csv at the specified file path in the output of annotations_parser.py

5. Download WFLW training images WFLW Training and Testing Images from https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view  --- for Google Colab, save WFLW_images.tar.gz to Google Drive, otherwise store locally or on chosen virtual environment

COLAB:
- Training code at https://colab.research.google.com/drive/1r_dXCJkMsFt6vrwoZXCks5hQXr6LpyDq
- Simply upload the saving_trainer.csv file into your runtime and allow for your Google Drive to be mounted when running the training.
- The model for the executed image attribute will be saved to a .keras file within your run time after all epochs are complete.
- Download the .keras file to save your model when satisfied with training results

TEST CASES:
- *Manipulate Epochs, Learning Rate, and Batch Size to view accuracies of different attributes*
- Function: run_5_times(csv_data, epoch, attribute) --- Compare the speed of program in 5 runs within one runtime VS 100 epochs in one run
- Function: run_all_fwd_rev(csv_data, epoch) --- Ensures that order of attributes listed does not effect rate of loss

# Additional Files:
> image_lighting_calc.py --- Script to find lighting values of input images and store them in a csv file
> Lighting-Trainer.ipynb --- Uses csv file of lightings developed from previous file to train a model to recognize new images' lighting values on a scale from 0-2
