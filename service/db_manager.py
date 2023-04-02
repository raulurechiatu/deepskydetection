import csv
import re
import time
from pathlib import Path

import main_ai

# from pandas import read_csv

db_location = 'resources/galaxyzoo2/data/'
csv_filename_mappings = 'gz2_filename_mapping.csv'
csv_db = 'gz2_hart16.csv'
filenames, filename_headers, db_data, db_data_headers = [], [], [], []

class_to_number_mapping = {
    'sc': 0,
    'sb': 1,
    'er': 2,
    'ei': 3,
    'ec': 4,
    'se': 5,
    'sd': 6,
    'a':  7,
    'sa': 8
}
# class_to_number_mapping = {
#     's': 0,
#     'e': 1,
#     'a':  2
# }


# A method used to search db for a filename
# This has as requirement the load_dbs method to be pre called because it depends on the variables it populates
def search_file(files):
    if len(filenames) == 0 or len(db_data) == 0:
        print("No data found in the data bases, please check if load_dbs method was run beforehand.")
        return

    before = time.time()
    results = []

    for file_data in files:
        # Gets the actual name of the file without the extension (100.jpg -> 100)
        final_name = file_data['filename'][:-4]
        filename_row = next((x for x in filenames if x[2] == final_name), None)
        result_row = next((x for x in db_data if x[0] == filename_row[0]), None)
        if result_row is not None:
            results.append(result_row)

    after = time.time()
    print("DB Search took", (after-before), "s and generated ", len(results), " valid results")

    return results


def get_data(file_names):
    before = time.time()

    file_mappings, csv_data = get_csv_raw()
    data = []
    skipped_files = []
    skip_reason = ""
    for file_name in file_names:
        file_name = file_name[:-4]
        # OBS. To remove a lot of the time needed for the search remove the next two lines
        # Get the object id from the file mappings to the data object id
        obj_id = next(file_mapping for file_mapping in file_mappings if file_name == file_mapping[2])[0]
        # Get the list of objects with the same id
        # obj_ids = list(filter(lambda file_mapping: file_mapping[0] == obj_id, file_mappings[int(file_name)-20:int(file_name)+20]))
        # If an object appears multiple times with the same id skip it
        # if len(obj_ids) > 1:
        #     continue
        # Get the index of the current object from all the objects with the same ID
        # obj_index = obj_ids.index(next(file_mapping for file_mapping in obj_ids if file_name == file_mapping[2]))
        try:
            # OBS. To remove a lot of the time needed for the search uncomment the next line and remove the following
            data_item = next(csv_item for csv_item in csv_data if obj_id == csv_item[0])
            # Get the same id of the object from the data list
            # data_item = list(filter(lambda csv_item: csv_item[0] == obj_id, csv_data))[obj_index]
        except Exception as e:
            skipped_files.append(file_name)
            skip_reason = e
        if data_item is not None:
            data.append(data_item)
        # else:
        #     main_ai.images_to_load -= 1
    after = time.time()
    print("Data mapping took", (after-before), "s for ", len(data), " valid results")
    print("Skipped ", skipped_files, " because of ", skip_reason)
    main_ai.images_to_load = len(data)

    return data


def is_data_valid(file_name, file_mappings, csv_data):
    file_name = file_name[:-4]
    # OBS. To remove a lot of the time needed for the search remove the next two lines
    # Get the object id from the file mappings to the data object id
    obj_id = next(file_mapping for file_mapping in file_mappings if file_name == file_mapping[2])[0]
    # Get the list of objects with the same id
    obj_ids = list(filter(lambda file_mapping: file_mapping[0] == obj_id, file_mappings[int(file_name) - 20:int(file_name) + 20]))
    # If an object appears multiple times with the same id skip it
    if len(obj_ids) > 1:
        return False
    # This part checks if the db with all the data has multiple entries with the same id (computational intense)
    # try:
    #     # Get the same id of the object from the data list
    #     data_items = list(filter(lambda csv_item: csv_item[0] == obj_id, csv_data))
    #     if len(data_items) > 1:
    #         return False
    # except Exception as e:
    #     print(e)
    return True


def get_csv_raw():
    mappings_path = Path(__file__).parent.parent / db_location / csv_filename_mappings
    file_mappings, _ = read_csv(mappings_path)
    data_path = Path(__file__).parent.parent / db_location / csv_db
    csv_data, csv_headers = read_csv(data_path)
    return file_mappings, csv_data


def get_labels(galaxy_data):
    labels = []
    before = time.time()
    for data in galaxy_data:
        labels.append(get_label_value(data))
    after = time.time()

    print("Labels were assigned to images successfully in " + str(after-before))
    return None, labels


def get_label_value(data):
    # column_offset should be 0, 1 or 2
    # this represents which value to take from the table, 0 is fraction, 1 is weighted fraction, 2 is debiased
    column_offset = 0
    if float(data[11 + column_offset]) >= 0.469 and float(data[101 + column_offset]) >= 0.5:
        return 0
    elif float(data[11 + column_offset]) >= 0.469 and float(data[107 + column_offset]) >= 0.5:
        return 1
    elif float(data[11 + column_offset]) >= 0.469 and float(data[113 + column_offset]) >= 0.5:
        return 2
    elif float(data[17 + column_offset]) >= 0.43 and float(data[29 + column_offset]) >= 0.602:
        return 3
    elif float(data[17 + column_offset]) >= 0.43 and float(data[35 + column_offset]) >= 0.715 and float(data[53 + column_offset]) >= 0.619:
        return 4
    else:
        return 5



def get_galaxy_classes(galaxy_data, rotations=4):
    labels = []
    indexed_labels = []
    for data in galaxy_data:
        for i in range(rotations):
            labels.append(data[6][:2])
        index = class_to_number_mapping[data[6][:2].lower()]
        for i in range(rotations):
            indexed_labels.append(index)
    # indexed_labels = get_uniques(labels)
    return labels, indexed_labels


def get_uniques(labels):
    indexed_labels = []
    list_set = set(labels)
    uniques = (list(list_set))
    for i in range(len(uniques)):
        if re.search(r'[^a-zA-Z]', uniques[i]) is None:
            continue
        res = re.search(r'[^a-zA-Z]', uniques[i]).start()
        uniques[i] = uniques[i][0: res]
    list_set = set(uniques)
    uniques = (list(list_set))

    print(uniques)
    for label in labels:
        res = label
        if re.search(r'[^a-zA-Z]', label) is not None:
            res = re.search(r'[^a-zA-Z]', label).start()
            res = label[0: res]
        indexed_labels.append(uniques.index(res))
    return indexed_labels


def class_to_number(labels):
    class_numbers = []
    for label in labels:
        class_numbers.append(class_to_number_mapping[label.lower()])
    return class_numbers


# A method used to save in memory what we have in the CSV dbs
# Done in order to quickly search for multiple data with only 1 operation of loading and saving the rows in memory
def load_dbs(files_to_load=-1):
    global filenames, filename_headers, db_data, db_data_headers
    path = Path(__file__).parent.parent / db_location / csv_filename_mappings
    filenames, filename_headers = read_csv(path, files_to_load)
    db_data, db_data_headers = read_csv(path, files_to_load)


# Used to open and read the csv files
def read_csv(path, files_to_load=-1):
    before = time.time()
    data = []
    with open(path) as db:
        reader = csv.reader(db, delimiter=',')

        if reader.__sizeof__() == 0:
            print("Invalid csv file.")
            return []

        # Get the header names in a list
        headers = next(reader)

        # A more efficient way to iterate over the whole dataset without adding a check step each time
        current_file = 0
        if files_to_load == -1:
            for row in reader:
                data.append(row)

        else:
            for row in reader:
                if current_file >= files_to_load:
                    break
                current_file += 1
                data.append(row)

    after = time.time()
    print("Loaded", files_to_load, "rows for csv file", path, "in", (after-before), "s")

    return data, headers
