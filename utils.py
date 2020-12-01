import os
import csv
from fnmatch import fnmatch


def find_doc_files_path(root, pattern="*.docx"):
    file_paths = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                file_paths.append(os.path.join(path, name))
    return file_paths


def create_csv(filename):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "Sentence", "Comment", "Class"])


def write_csv(sent_comm_dict, all_sentences, filename, id, only_with_comment=False):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for sentence in all_sentences:
            label = '1' if (sentence in sent_comm_dict) else '0'
            if label == '0' and only_with_comment is False:
                writer.writerow([id, sentence, "None", label])
            elif label == '1':
                writer.writerow([id, sentence, sent_comm_dict[sentence][0], label])
            id = id + 1
        return id
