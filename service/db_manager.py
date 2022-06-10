import csv
import time

db_location = 'resources/galaxyzoo2/csv/'
csv_filename_mappings = 'gz2_filename_mapping.csv'
csv_db = 'gz2_hart16.csv'
filenames, filename_headers, db_data, db_data_headers = [], [], [], []


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


# A method used to save in memory what we have in the CSV dbs
# Done in order to quickly search for multiple data with only 1 operation of loading and saving the rows in memory
def load_dbs(files_to_load=-1):
    global filenames, filename_headers, db_data, db_data_headers
    filenames, filename_headers = read_csv(db_location + csv_filename_mappings, files_to_load)
    db_data, db_data_headers = read_csv(db_location + csv_db, files_to_load)


# Used to open and read the csv files
def read_csv(path, files_to_load):
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
                current_file += 1
                data.append(row)

        else:
            for row in reader:
                if current_file > files_to_load:
                    break
                current_file += 1
                data.append(row)

    after = time.time()
    print("Loaded", files_to_load, "rows for csv file", path, "in", (after-before), "s")

    return data, headers
