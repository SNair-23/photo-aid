# Photo-aid
A deep-learning based application that recommends changes to portraits based off of the factors of blur, occlusion, make up, illumination, expression, and pose. Ideal for headshot photography. Instructions below are written for use with Google Colab for running models using their pre-existing libraries of Tensorflow, Keras, Matplotlib, Numpy, and OpenCV.

# Prerequisites
- Linux OS
- Python 3
- Pandas Library (Local- to run annotations_parser.py)

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
*Create a copy of https://colab.research.google.com/drive/1ErgGkizOYxa2sGaNvckm0tb2Bksu7fZm?usp=sharing to manipulate and run the testing code
*CONSTANTS: tar_path, extract_path, image_dir, model_path, illumination_csv_data, and image_dir2 are constants in the main() function of the NeuralNet_Tester_Colab.py program. Please ensure to manipulate them according to your file locations in drive. 

  illumination_csv_data = pd.read_csv("/content/drive/MyDrive/image-lightings.csv")
  image_dir2 = '/content/drive/MyDrive/all-images'
  
1. Model Can be tested on WFLW Testing data. Again run:
   ~~~
   python3 ~./Downloads/annotations_parser.py
   ~~~
   (Type "E" to load testing annotations)
   - Upload the csv derived from annotations_parser.py as the input csv file to evaluate the model of each attribute.
   - Ensure that the desired .keras models for all attributes are uploaded to Google Drive, altering the filepath as needed.
   - The main() function will run and evaluate all the models on the WFLW testing data when run
   

2. You can also test the accuracy of the model for illumination using your own image files.
   - Download the "image_lighting_calc.py" file and assign the absolute path to your Google Drive folder of images to "image_folder".
   - Upload the csv created from the image-lighting-calc.py script to Google Drive
   - In function main() of the tester code, change the illumination_csv_data constant to the absolute path of your csv
   - my_imgs_tester() should now evaluate the model's performance on your annotated images when run

