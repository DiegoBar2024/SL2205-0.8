from email import header
import pandas
import os
from statistics import mean
import csv
import numpy as np

def main():
    #Path to sampleid with used ids
    id_path = "/home/sere/seretest/clasificaciones_antropometricos.csv"
    #Path to samples you want to verify
    folders_path = "/home/sere/seretest/raw_2_process"

    new_path = '/home/sere/seretest/classification.txt'

    #range for acceleration
    ac_min = 8
    ac_max = 12


    in_range_postive, in_range_negative, not_in_range, no_sample, wrong_column=classification0(id_path, folders_path, ac_min, ac_max)
    with open(new_path, 'w') as file:
        file.write(f"In range positive: {in_range_postive} \n")
        file.write(f"In range negative: {in_range_negative} \n")
        file.write(f"Not in range: {not_in_range} \n")
        file.write(f"No sample found: {no_sample}\n")
        file.write(f"Wrong column: {wrong_column}\n")


#returns avarage acceleration and number of the folder (S+id)
def average_ac2(folders_path,id):
    ac_y_lists = list()
    ac_y =list()

    #Reads files in raw_2_process and calculaes average acceleration for samples with
    #id in sample_id
    for folder in os.listdir(folders_path):
        if folder == f"S{id}":
            #print(f"{folders_path}/{folder}/3{folder}.csv")
            #Check where the column of the acceleration
            csv_columns = pandas.read_csv(folders_path+"/"+folder+"/3"+folder+".csv", usecols=[2, 3], nrows= 3, sep="\t")
            if csv_columns.iloc[1,1] == "CAL":
                n=3
            else:
                n=2
            
            csv_data = pandas.read_csv(folders_path+"/"+folder+"/3"+folder+".csv", usecols=[n],skiprows= [0,1,2,3], nrows= 15000, sep="\t")
            #print( "csv_data",csv_data)
            #this makes a list, that contains list range[1] with the value i want
            ac_y_lists = csv_data.values.tolist()
            #print(ac_y_lists)
            #move everything to another list that only contains float values
            for value in ac_y_lists:
                ac_y.append(float(value[0]))
            #print(mean(ac_y))
            return mean(ac_y), folder

#clasifies acceleration of samples and returns the name of the sample in 
#list of in range postivie (acceleration is in the gicen range and is positive),
#in range negative and not in range.
def classification0(id_path, folders_path, ac_min, ac_max):

    #Reads id we're going to use
    csv_id = pandas.read_csv(id_path, skiprows=[0], usecols=[0])
    sample_id = csv_id.values.tolist()

    in_range_postive = list()
    in_range_negative = list()
    not_in_range = list()
    no_sample = list()
    wrong_column = list()

    for id in sample_id:
        #print(f"S{id[0]}")
        #print("class", average_ac2(folders_path,id))

        #checks if the sample is there and makes avarege ac
        if os.path.exists(folders_path+"/"+f"S{id[0]}"+"/3"+f"S{id[0]}"+".csv"):
            average_acceleration, sample_id = average_ac2(folders_path,id[0])
            #print(average_acceleration, sample_id)
            #verifies if the aceleration is in range and its positive and adds sample id to list
            if ac_min <= average_acceleration <= ac_max: #in range(ac_min, ac_max):
                in_range_postive.append(id[0])
            elif -ac_max <= average_acceleration <=-ac_min: #in range(- ac_max, -ac_min):
                in_range_negative.append(id[0])
            elif average_acceleration > 500:
                wrong_column.append(id[0])
            else:
                not_in_range.append(id[0])
            
        else:
            no_sample.append(id[0])
            #print("here", id)
    #print(in_range_postive, in_range_negative, not_in_range, no_sample)
    return in_range_postive, in_range_negative, not_in_range, no_sample, wrong_column
#returns avarage acceleration and standard desviation)
def average_ac3(folders_path,id):
    ac_y_lists = list()
    ac_y =list()

    #Reads files in raw_2_process and calculaes average acceleration for samples with
    #id in sample_id
    for folder in os.listdir(folders_path):
        if folder == f"S{id}":
            #print(f"{folders_path}/{folder}/3{folder}.csv")
            #Check where the column of the acceleration
            csv_columns = pandas.read_csv(folders_path+"/"+folder+"/3"+folder+".csv", usecols=[2, 3], nrows= 3, sep="\t")
            if csv_columns.iloc[1,1] == "CAL":
                n=3
            else:
                n=2
            
            csv_data = pandas.read_csv(folders_path+"/"+folder+"/3"+folder+".csv", usecols=[n],skiprows= [0,1,2,3], nrows= 15000, sep="\t")
            #print( "csv_data",csv_data)
            #this makes a list, that contains list range[1] with the value i want
            ac_y_lists = csv_data.values.tolist()
            #print(ac_y_lists)
            #move everything to another list that only contains float values
            for value in ac_y_lists:
                ac_y.append(float(value[0]))
            #print(mean(ac_y))
            return mean(ac_y), np.std(ac_y)

#main()