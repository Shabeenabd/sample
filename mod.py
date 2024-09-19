# import OS module
import os
# Get the list of all files and directories
path = "/home/binu/data/10Classes_Aug1"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# prints all files
print(dir_list)

