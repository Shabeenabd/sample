
import os
import numpy as np
import shutil
import random
#import torch
#import torchvision
#from torch import nn
#from torchvision import transforms

dir_path = "/home/shabeen/project/model_codes/NEW/"

res = os.listdir(dir_path)
print(res)

# root_dir = "/home/binu/data/24_class_aug17/"
new_root = 'AllDatasets/'
#classes = ['Chathuramulla', 'Cherukadaladi', 'Arayal', 'Athi', 'Ayyappana', 'Aeriku', 'Anadaman thipali', 'Grambu', 'Aanachuvadi', 'Amukram']
#classes = ['Chathuramulla', 'Cherukadaladi', 'Arayal', 'Athi', 'Ayyappana', 'Aeriku', 'Naruneendi', 'Anadaman thipali', 'Aanachuvadi', 'Amukram']
# classes = ['Chathuramulla', 'Karimkurinji', 'Cherukadaladi', 'Uzhinja', 'Arayal', 'Athi', 'Ayyappana', 'Thipali', 'Shavakotta_pacha', 'Vellakoduveli', 'Kasthoorivenda', 'Thetti', 'Pushkaramulla', 'Aeriku', 'arootha', 'Murikooti', 'Naruneendi', 'Anadaman thipali', 'Cheruthekku', 'NeelaUmmam', 'Aanachuvadi', 'Neelakoduveli', 'Chittamruthu', 'Amukram']
classes=res
#classes = ['Karimkurinji', 'Vellakoduveli', 'Kasthoorivenda', 'Thetti', 'Pushkaramulla', 'Murikooti', 'Cheruthekku', 'NeelaUmmam']
#classes =['Karimkurinji', 'Uzhinja', 'Thipali', 'Shavakotta_pacha', 'Vellakoduveli', 'Kasthoorivenda', 'Thetti', 'Pushkaramulla', 'arootha', 'Murikooti', 'Cheruthekku', 'NeelaUmmam', 'Neelakoduveli', 'Chittamruthu']
for cls in classes:
	os.makedirs(dir_path + new_root + 'train/' + cls, exist_ok=True)
	os.makedirs(dir_path +new_root + 'test/' + cls, exist_ok=True)

for cls in classes:
	src= dir_path + cls
	print(src)

	allFileNames = os.listdir(src)
	np.random.shuffle(allFileNames)

	test_FileNames,train_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.25)])
	test_FileNames = [src+'/' + name for name in test_FileNames]
	train_FileNames = [src+'/' + name for name in train_FileNames]

	print('Total images : '+cls +'' +str(len(allFileNames)))
	print('Training : '+ cls + ''+str(len(train_FileNames)))
	print('Testing :'+ cls + ''+str(len(test_FileNames)))

	for name in train_FileNames:
		shutil.move(name, dir_path + new_root+'train/'+cls )
	for name in test_FileNames:
		shutil.move(name,dir_path + new_root+'test/'+cls)
