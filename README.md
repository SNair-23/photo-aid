# Photo-aid
A ongoing deep-learning based application that aims to recommend changes to portraits based off of the factors of blur, occlusion, make up, illumination, expression, and pose. Ideal for headshot photography. Instructions below are written for best use with Google Colab for running models using their pre-existing libraries of Tensorflow, Keras, Matplotlib, Numpy, and OpenCV.

# Prerequisites
- Linux OS
- Python 3.8-3.11
- Pandas Library (Local- to run annotations_parser.py)
- Google Colab is highly recommended to access pre-installed libraries (Tensorflow, Keras, Matplotlib, Numpy, OpenCV, Sklearn, etc)

# Loading Data
Photo-aid currently is trained with the intent of clasifying images based on 6 binary image attributes labeled in the Wider Facial Landmarks in-the-wild (WFLW) dataset. Complete information on the WFLW dataset can be found at https://wywu.github.io/projects/LAB/WFLW.html 

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
   
# Model Training 
IF USING GOOGLE COLAB (No need to install necessary packages, and free limited access to GPU acceleration):
- Training code at https://colab.research.google.com/drive/1r_dXCJkMsFt6vrwoZXCks5hQXr6LpyDq?usp=sharing --- Create a copy to edit
- CONSTANTS: tar_path, extract_path, image_dir, and csv_path are constants in the main() function of the NeuralNet_Tester_Colab.py program. --- Please ensure to manipulate them according to your file locations in drive.
- The model for the executed image attribute will be outputed to a .keras file within your run time after all epochs are complete.
- Download the .keras file to save your model when satisfied with training results

TRIALS:
- *Manipulate Epochs, Learning Rate, and Batch Size to view accuracies of different attributes*
- Function: run_5_times(csv_data, epoch, attribute) --- Compare the speed of program in 5 runs with 20 epochs VS 1 run of 100 epochs
- Function: run_all_fwd_rev(csv_data, epoch) --- Ensures that order of attributes listed does not effect rate of loss

# Model Testing
*Create a copy of https://colab.research.google.com/drive/1ErgGkizOYxa2sGaNvckm0tb2Bksu7fZm?usp=sharing to manipulate and run the testing code

*CONSTANTS: tar_path, extract_path, image_dir, model_path, illumination_csv_data, and image_dir2 are constants in the main() function of the NeuralNet_Tester_Colab.py program. Please ensure to manipulate them according to your file locations in drive. 
  
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

# Running it locally (Less reliable)
Devises with GPU and CUDA Compute Capability are recommended to accelerate training, but it can still be run (slowly) on CPU. 

1. Create Virtual Evironment (recommended):
~~~
python -m venv venv
source venv/bin/activate
~~~
2. Install the following libraries.
~~~
pip install tensorflow
pip install numpy
pip install matplotlib
pip install pandas
pip install opencv-python
pip install scikit-learn
~~~
3. Clone this Repository
~~~
git clone git@github.com:SNair-23/photo-aid.git
~~~
4. EDITS TO FILES:
  - wflw_nn_trainer_colab.py & neuralnet_tester_colab.py: Comment out lines importing google colab and mounting google drive. Change constants in main() to match absolute paths to the files stored in your local directory.
  - image_lighting_calc.py: Comment out lines importing google colab and mounting google drive. Change image_folder to the absolute path to your local folder of images that you want to script. 
  - make sure to keep track of file locations created in image_lighting_calc.py(image-lightings.csv) and wflw_nn_trainer_colab.py(.keras models) to parse into neuralnet_tester_colab.py.


# Results
Results of the training and testing have been decribed in the Paper- instructions to compile are below. 

# Paper
To compile the paper, clone the repository (if you haven't already) with:
~~~
git clone git@github.com:SNair-23/photo-aid.git
~~~
Then run the following commands:
~~~
cd photo-aid/ShriyaNair_PortraitOptimizationviaDeepLearning/
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
~~~
Open a pdf version of the paper using:
~~~
xdg-open main.pdf
~~~

# Graphs
Here is how to replicate the specific plots seen in the paper (png files of plots can be found in the ShriyaNair_PortraitOptimizationviaDeepLearning/ folder above):

- 15-epoch-fvr3.png: run the function run_all_fwd_rev(csv_data, image_dir, 15) in wflw_nn_trainer_colab.py
- BLUR1.png: run the function run_and_plot(csv_data, image_dir, 'blur', epoch=100, lr=0.0028) in wflw_nn_trainer_colab.py
- Distro-of-binary-annotation-labels.png: uncomment the #Test Cases below the plot_csv_distribution() function in wflw_nn_trainer_colab.py
- blururur.png: run the function run_5_times(csv_data, image_dir, 'blur', 20) in wflw_nn_trainer_colab.py

Results may not be identical to those viewed within this project because they vary on each training run. Feel free to experiment with the parameters given above to view different results.
