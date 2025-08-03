This is the paper report of the outcomes of the initial stages of Photo-aid. To compile the paper, do the following steps (also described in the "Paper" section of photo-aid/README.md):

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
Below is the duplicate of the "Graphs" section of photo-aid/README.md

Here is how to replicate the specific plots seen in the paper (png files of plots can be found in the ShriyaNair_PortraitOptimizationviaDeepLearning/ folder above):

    15-epoch-fvr3.png: run the function run_all_fwd_rev(csv_data, image_dir, 15) in wflw_nn_trainer_colab.py
    BLUR1.png: run the function run_and_plot(csv_data, image_dir, 'blur', epoch=100, lr=0.0028) in wflw_nn_trainer_colab.py
    Distro-of-binary-annotation-labels.png: uncomment the #Test Cases below the plot_csv_distribution() function in wflw_nn_trainer_colab.py
    blururur.png: run the function run_5_times(csv_data, image_dir, 'blur', 20) in wflw_nn_trainer_colab.py

Results may not be identical to those viewed within this project because they vary on each training run. Feel free to experiment with the parameters given above to view different results.
