#! /usr/bin/env python3

import time, os, json
import pandas as pd
import sys

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
        
def parseAnnotationLine(data_row):
    
    data_dictionary = {}
    tokens = data_row.split()
    data_dictionary["coordinates"] = tokens[0:-11]
    data_dictionary["file_name"] = tokens[-1]
    data_dictionary["blur"] = tokens[-2]
    data_dictionary["occlusion"] = tokens[-3]
    data_dictionary["make_up"] = tokens[-4]
    data_dictionary["illumination"] = tokens[-]5
    data_dictionary["expression"] = tokens[-6]
    data_dictionary["pose"] = tokens[-7]
    data_dictionary["y_max"] = tokens[-8]
    data_dictionary["x_max"] = tokens[-9]
    data_dictionary["y_min"] = tokens[-10]
    data_dictionary["x_min"] = tokens[-11]
    
    return data_dictionary



def read_as_csv(list_of_dicts, output_file):
  import csv

  with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=list_of_dicts[0].keys())
    writer.writeheader()
    writer.writerows(list_of_dicts)

  csv_data = pd.read_csv(output_file)
  return csv_data


def main():
    print("~~~~~~~~~~\n")
    print("~ SCRIPT ~\n")
    print("~~~~~~~~~~\n")
    question = input("Enter R for training or E for testing: ")
    if question == 'R':
        r_or_e =  "/list_98pt_rect_attr_train.txt"
        csv_path = "/saving_trainer.csv"
    elif question == 'E':
        r_or_e = "/list_98pt_rect_attr_test.txt"
        csv_path = "/saving_tester.csv"
    annotations_set = {}
    annotations_path = os.path.abspath('./Downloads/WFLW_files/datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/')                                
    annotations = open(annotations_path + r_or_e)
    recording_file_path = annotations_path + csv_path
    recording_file = open(recording_file_path, "w")
    list_of_dicts = []
    for data_point in annotations:  #for each row in annotations
        parsed_data_point = parseAnnotationLine(data_point) #return a dictionary 
        annotations_set[parsed_data_point["file_name"]] = parsed_data_point
        list_of_dicts.append(parsed_data_point)
    read_as_csv(list_of_dicts, recording_file_path)
                
    recording_file.close()
    annotations.close()

    print("Your new csv file has been succesfully stored to " + recording_file_path)
    return recording_file

  

"""
        print("Pick A WFLW Image Folder and Image to get annotations for.\n")
        print("\tIE: '37--Soccer/37_Soccer_soccer_ball_37_45.jpg'\n")
        print("To Exit enter 'quit'\n")
        choice = input("Enter your choice: ")

        if choice == 'quit':
            print("Exiting the Terminal. Goodbye!")
            break
        else:
            print(annotations_set[choice])
"""

if __name__ == "__main__":
    main()
