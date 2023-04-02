import time

from numpy import delete


def remove_class(data, labels, label_to_remove):
    before = time.time()
    indexes_to_remove = []
    for index in range(len(data)):
        if labels[index] == label_to_remove:
            indexes_to_remove.append(index)

    for remove_index in sorted(indexes_to_remove, reverse=True):
        data = delete(data, remove_index, 0)
        del labels[remove_index]

    after = time.time()
    print("Removed the last class (other) in " + str(after-before))
    return data, labels
