import glob
import pandas as pd

# Define the folder path and the file extension
folder_path = "/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/Result/DAN/Byte_512/"
file_extension = ".csv"

# Create an empty list to store the file names
csv_files = []

# Loop through the folder and its subfolders and find all csv files
for file in glob.glob(folder_path + "**/*" + file_extension, recursive=True):
    # Append the file name to the list
    csv_files.append(file)

# Loop through the list of csv files and read them into a DataFrame
df = pd.concat([pd.read_csv(f) for f in csv_files])

# Group by the columns and calculate the mean
merge_df = df.groupby(['backbone', 'method', 'test_function', 'scenario', 'target', 'byte_size', 'trade_off', 'epoch']).mean()

# Save the merged DataFrame as a csv file
merge_df.to_csv("merged_results.csv")
