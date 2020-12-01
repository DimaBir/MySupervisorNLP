import os
from fnmatch import fnmatch


def find_doc_files_path(root=r"C:\Users\dmitry.v\PycharmProjects\MySupervisorUtils\data", pattern="*.docx"):
    file_paths = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                file_paths.append(os.path.join(path, name))
    return file_paths
