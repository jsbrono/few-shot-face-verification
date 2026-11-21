import os
import glob
from torchvision.transforms import transforms
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
import random

def create_index_map(filepath, num_refs=3):
    index_map = {}
    index = 0
    for image_path in os.listdir(filepath):

        true_dir = os.path.join(filepath, image_path)
        false_dir = os.path.join(filepath, image_path + "_forg")

        true_image_locations = glob.glob(os.path.join(true_dir, "*.png"))
        false_image_locations = glob.glob(os.path.join(false_dir, "*.png"))

        for image in true_image_locations:
            if len(true_image_locations) <= num_refs:
                continue
            reference_image_locations = np.random.choice(
                [
                    image_location
                    for image_location in true_image_locations
                    if image_location != image
                ],
                size=num_refs,
                replace=False,
            )
            index_map[index] = (reference_image_locations, image, 1)
            index += 1

        for image in false_image_locations:
            if len(true_image_locations) <= num_refs:
                continue
            reference_image_locations = np.random.choice(
                true_image_locations, size=num_refs, replace=False
            )
            index_map[index] = (reference_image_locations, image, 0)
            index += 1

        mapping_list = list(index_map.values())
        np.random.shuffle(mapping_list)
        index_map = {idx: val for idx, val in enumerate(mapping_list)}
    return index_map

class SignatureDataset(Dataset):
    def __init__(self, filepath):
        self.index_map = create_index_map(filepath)
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((155, 220)),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def __len__(self):
        return len(self.index_map.keys())

    def __getitem__(self, index: int):
        reference_locations, query_location, label = self.index_map[index]

        reference_data = torch.cat(
            [self.transform(read_image(ref_loc)) for ref_loc in reference_locations]
        )
        query_data = self.transform(read_image(query_location))
        label = torch.tensor(label, dtype=torch.float32)

        return reference_data, query_data, label

def create_dynamic_index_map(filepath):
    # Can extend training such that it uses randomly sized subsets of ref set
    index_map = {}
    index = 0
    max_refs = 0

    for image_path in os.listdir(filepath):
        true_dir = os.path.join(filepath, image_path)
        false_dir = os.path.join(filepath, image_path + "_forg")

        true_image_locations = glob.glob(os.path.join(true_dir, "*.png"))
        if len(true_image_locations) > max_refs:
            max_refs = len(true_image_locations)

    for image_path in os.listdir(filepath):
        true_dir = os.path.join(filepath, image_path)
        false_dir = os.path.join(filepath, image_path + "_forg")

        true_image_locations = glob.glob(os.path.join(true_dir, "*.png"))
        false_image_locations = glob.glob(os.path.join(false_dir, "*.png"))

        for i in range(len(true_image_locations)):
            query_image_location = true_image_locations[i]
            reference_image_locations = [true_image_locations[j] for j in range(len(true_image_locations)) if j!=i]
            mask = [1] * len(reference_image_locations)
            num_missing = max_refs - len(reference_image_locations)
            reference_image_locations += [None] * num_missing
            mask += [0] * num_missing

            index_map[index] = (reference_image_locations,query_image_location,1,mask)
            index+=1

        for false_image_location in false_image_locations:
            reference_image_locations = true_image_locations[:]

            mask = [1] * len(reference_image_locations)
            num_missing = max_refs - len(reference_image_locations)
            reference_image_locations += [None] * num_missing
            mask += [0] * num_missing
            
            index_map[index] = (reference_image_locations, false_image_location, 0, mask)
            index += 1

    mapping_list = list(index_map.values())
    np.random.shuffle(mapping_list)
    index_map = {idx: val for idx, val in enumerate(mapping_list)}
    return index_map

class SignatureDynamicDataset(Dataset):
    def __init__(self, filepath):
        self.index_map = create_dynamic_index_map(filepath)
        self.transform = transforms.Compose(
            [
                # transforms.Grayscale(),
                transforms.Resize((155, 220)),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def __len__(self):
        return len(self.index_map.keys())

    def __getitem__(self, index: int):
        reference_locations, query_location, label, mask = self.index_map[index]

        query_data = self.transform(read_image(query_location))

        reference_data = []
        for ref_loc in reference_locations:
            if ref_loc is None:
                ref_tensor = torch.zeros_like(query_data)
            else:
                ref_tensor = self.transform(read_image(ref_loc))
            reference_data.append(ref_tensor)

        reference_data = torch.stack(reference_data,dim=0)

        label = torch.tensor(label, dtype=torch.float32)
        mask = torch.tensor(mask,dtype= torch.float32)

        return reference_data, query_data, label, mask
    
def create_index_map_lfw(image_path,matches_csv,mismatches_csv):
    index_map = {}
    idx = 0
    for _,row in matches_csv.iterrows():
        name,image1_idx,image2_idx = row
        name_path = os.path.join(image_path,name)
        image1_location = os.path.join(name_path,os.listdir(name_path)[image1_idx-1])# not actually index, just number in filename
        image2_location = os.path.join(name_path,os.listdir(name_path)[image2_idx-1])
        index_map[idx] = (image1_location,image2_location,1)
        idx+=1

    for _,row in mismatches_csv.iterrows():
        name1,image1_idx,name2,image2_idx = row
        name1_path = os.path.join(image_path,name1)
        name2_path = os.path.join(image_path,name2)
        image1_location = os.path.join(name1_path,os.listdir(name1_path)[image1_idx-1])
        image2_location = os.path.join(name2_path,os.listdir(name2_path)[image2_idx-1])
        index_map[idx] = (image1_location,image2_location,0)
        idx+=1

    return index_map

class LFWDataset(Dataset):
    def __init__(self, filepath,train=True):
        image_path = os.path.join(filepath, "lfw-deepfunneled", "lfw-deepfunneled")

        if train:
            matches_csv = pd.read_csv(os.path.join(filepath, "matchpairsDevTrain.csv"))
            mismatches_csv = pd.read_csv(os.path.join(filepath, "mismatchpairsDevTrain.csv"))
        else:
            matches_csv = pd.read_csv(os.path.join(filepath, "matchpairsDevTest.csv"))
            mismatches_csv = pd.read_csv(os.path.join(filepath, "mismatchpairsDevTest.csv"))

        self.num_positives = len(matches_csv)
        self.num_negatives = len(mismatches_csv)
        
        self.index_map = create_index_map_lfw(image_path,matches_csv,mismatches_csv)
        normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        if train == True:
            self.transforms = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float32),
                    normalise,
                    transforms.Resize((250, 250)),
                    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
                    # transforms.RandomHorizontalFlip(),                   
                    # transforms.RandomVerticalFlip(p=0.2),                 
                    # transforms.RandomRotation(15),                        
                    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
                    # transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)), 
                    # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  
                    transforms.ConvertImageDtype(torch.float32),
                ]
            )        
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((250, 250)),
                    transforms.ConvertImageDtype(torch.float32),
                    normalise
                ]
            )        

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index: int):
        query_location,reference_location,label = self.index_map[index]

        reference_data = self.transforms(read_image(reference_location)).unsqueeze(0)
        query_data = self.transforms(read_image(query_location))
        label = torch.tensor(label, dtype=torch.float32)
        mask = torch.ones(reference_data.size(0), dtype=torch.float32)
        return reference_data,query_data,label,mask
    
def create_index_map_casia(image_path,max_refs=3):
    index_map = {}
    index = 0
    num_positives = 0
    num_negatives = 0
    persons_folders = os.listdir(image_path)

    for person in persons_folders:
        # doing small case for now
        persons_image_path = [os.path.join(image_path, person, img) for img in os.listdir(os.path.join(image_path, person))]

        positive_sample = random.choice(persons_image_path)

        reference_image_locations = random.sample([img for img in persons_image_path if img!=positive_sample],k=min(max_refs,len(persons_image_path)-1))

        negative_person = random.choice([p for p in persons_folders if p != person])
        neg_person_images = [os.path.join(image_path, negative_person, img) for img in os.listdir(os.path.join(image_path, negative_person))]
        negative_sample = random.choice(neg_person_images)

        mask = [1] * len(reference_image_locations)
        num_missing = max_refs - len(reference_image_locations)
        reference_image_locations += [None] * num_missing
        mask += [0] * num_missing

        index_map[index] = (reference_image_locations,positive_sample,1,mask)
        index+=1
        num_positives+=1

        index_map[index] = (reference_image_locations,negative_sample,0,mask)
        index+=1
        num_negatives+=1

    mapping_list = list(index_map.values())
    random.shuffle(mapping_list)
    index_map = {i: v for i, v in enumerate(mapping_list)}
    return index_map, num_positives, num_negatives


class CasiaDataset(Dataset):
    def __init__(self, filepath,train=True):
        image_path = os.path.join(filepath, "casia-webface")

        self.index_map,self.num_positives,self.num_negatives = create_index_map_casia(image_path)
        # normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225])
        if train == True:
            self.transforms = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float32),
                    # normalise,
                    transforms.Resize((250, 250)),
                    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
                    # transforms.RandomHorizontalFlip(),                   
                    # transforms.RandomVerticalFlip(p=0.2),                 
                    # transforms.RandomRotation(15),                        
                    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
                    # transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)), 
                    # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  
                    transforms.ConvertImageDtype(torch.float32),
                ]
            )        
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((250, 250)),
                    transforms.ConvertImageDtype(torch.float32),
                    # normalise
                ]
            )        

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index: int):
        reference_locations, query_location, label, mask = self.index_map[index]

        query_data = self.transforms(read_image(query_location))

        reference_data = []
        for ref_loc in reference_locations:
            if ref_loc is None:
                ref_tensor = torch.zeros_like(query_data)
            else:
                ref_tensor = self.transforms(read_image(ref_loc))
            reference_data.append(ref_tensor)

        reference_data = torch.stack(reference_data,dim=0)

        label = torch.tensor(label, dtype=torch.float32)
        mask = torch.tensor(mask,dtype= torch.float32)

        return reference_data,query_data,label,mask
  