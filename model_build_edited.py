#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
#copyright aug 29 2024 icfoss
#@author shabeen 

#hyper parameters 
path = "/home/shabeen/project/model_codes/NEW/AllDatasets"
epoch=5
optimizer_name='ad'
batch_size=32
lr=1e-3

import wandb
import os
import torch
from tqdm.auto import tqdm
import torchvision
from typing import List, Dict, Tuple
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#import PIL to resolve image truncated error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datetime import datetime
now = datetime.now()
date = now.strftime('%b%d')

dir_list = os.listdir(os.path.join(path,'train'))
total_class=len(dir_list)
run_name=f"{total_class}cls_epch{str(epoch)+'_'+date+'_'+optimizer_name}bsze{batch_size}"


def accuracy_fn(y_true, y_pred):
	correct = torch.eq(y_true, y_pred).sum().item()
	acc = (correct / len(y_pred)) * 100
	return acc

def print_train_time(start, end, device=None):
	total_time = end - start
	print(f"\nTrain time on {device}: {total_time:.3f} seconds")
	return total_time

def set_seeds(seed: int=42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
device


wandb.init(project='medicinal_plant_classifier', name=run_name,
		   config={
    "learning_rate": lr,
    "epochs": epoch,
    })
# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

#change the classifier head �
# Get the list of all files and directories
class_names=dir_list
set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
# pretrained_vit # uncomment for model output

# from torchinfo import summary
# summary(model=pretrained_vit,input_size=(32, 3, 224, 224),col_names=["input_size", "output_size", "num_params", "trainable"],col_width=20,row_settings=["var_names"])

train_dir=os.path.join(path,'train')
test_dir=os.path.join(path,'test')

pretrained_vit_transforms = pretrained_vit_weights.transforms()

NUM_WORKERS =  4
def create_dataloaders(
	train_dir: str,
	test_dir:str,
	transform: transforms.Compose,
	batch_size: int,
	num_workers: int=NUM_WORKERS
	):
	train_data = datasets.ImageFolder(train_dir, transform=transform)
	test_data = datasets.ImageFolder(test_dir, transform=transform)

	class_names = train_data.classes
	print('Classes :',class_names)
	train_dataloader = DataLoader(
		train_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
	)
	test_dataloader = DataLoader(
		test_data,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)
	return train_dataloader, test_dataloader, class_names

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
		model.train()
		train_loss, train_acc = 0, 0
		for batch, (X, y) in enumerate(dataloader):
			X, y = X.to(device), y.to(device)
			y_pred = model(X)
			loss = loss_fn(y_pred, y)
			train_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
			train_acc += (y_pred_class == y).sum().item()/len(y_pred)

		train_loss = train_loss / len(dataloader)
		train_acc = train_acc / len(dataloader)
		return train_loss, train_acc

def test_step(model: torch.nn.Module,
		dataloader: torch.utils.data.DataLoader,
		loss_fn: torch.nn.Module,
		device: torch.device) -> Tuple[float, float]:
		model.eval()
		test_loss, test_acc = 0, 0
		with torch.inference_mode():
			for batch, (X, y) in enumerate(dataloader):
				X, y = X.to(device), y.to(device)
		
				test_pred_logits = model(X)

				loss = loss_fn(test_pred_logits, y)
				test_loss += loss.item()
				test_pred_labels = test_pred_logits.argmax(dim=1)
				test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
		test_loss = test_loss / len(dataloader)
		test_acc = test_acc / len(dataloader)
		return test_loss, test_acc
def enginetrain(model: torch.nn.Module,
		train_dataloader:torch.utils.data.DataLoader,
		test_dataloader:torch.utils.data.DataLoader,
		optimizer:torch.optim.Optimizer,
		loss_fn:torch.nn.Module,
		epochs: int,
		device:torch.device) -> Dict[str, List]:
		
		results = {"train_loss": [],
			"train_acc": [],
			"test_loss":[],
			"test_acc":[]}

		model.to(device)
		for epoch in tqdm(range(epochs)):
			train_loss, train_acc = train_step(model=model,
							dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer,device=device)
			test_loss, test_acc = test_step(model=model,dataloader=test_dataloader,loss_fn=loss_fn,device=device)

			print(
			f"Epoch: {epoch+1} | "
			f"train_loss: {train_loss:.4f} | "
			f"train_acc: {train_acc:.4f} | " 
			f"test_loss: {test_loss:.4f} | "
			f"test_acc: {test_acc:.4f}"
		)

			wandb.log({"train_loss": train_loss,
                   "train_acc": train_acc,
                   "test_loss": test_loss,
                   "test_acc": test_acc})	

		results["train_loss"].append(train_loss)
		results["train_acc"].append(train_acc)
		results["test_loss"].append(test_loss)
		results["test_acc"].append(test_acc)
		return results


train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                                     test_dir=test_dir,
                                                                                                     transform=pretrained_vit_transforms,
                                                                                                     batch_size=batch_size) # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)
optimizer = torch.optim.AdamW(params=pretrained_vit.parameters(),lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()
set_seeds()
pretrained_vit_results = enginetrain(model=pretrained_vit,train_dataloader=train_dataloader_pretrained,test_dataloader=test_dataloader_pretrained,optimizer=optimizer,loss_fn=loss_fn,epochs=epoch,device=device)

model_name=f"pretrain{total_class}cls_epch{str(epoch)+date+optimizer_name}bsze{batch_size}.pt"
torch.save(pretrained_vit, f"/home/shabeen/project/{model_name}")

wandb.finish()
