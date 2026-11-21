import kagglehub
from datasets.datasets import SignatureDataset,SignatureDynamicDataset, LFWDataset, CasiaDataset
from torch.utils.data import DataLoader

def make_signature_dataloaders(batch_size=256,shuffle=True):
    path = kagglehub.dataset_download("robinreni/signature-verification-dataset")
    train_image_path = path + "/sign_data/train"
    test_image_path = path + "/sign_data/test"

    train_dataset = SignatureDataset(train_image_path)
    test_dataset = SignatureDataset(test_image_path)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return train_loader,test_loader

def make_dynamic_signature_dataloaders(batch_size=256,shuffle=True,num_workers:int = 0):
    path = kagglehub.dataset_download("robinreni/signature-verification-dataset")
    train_image_path = path + "/sign_data/train"
    test_image_path = path + "/sign_data/test"

    train_dataset = SignatureDynamicDataset(train_image_path)
    test_dataset = SignatureDynamicDataset(test_image_path)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers
    )
    return train_loader,test_loader

def make_lfw_dataloaders(batch_size=256,shuffle=True,num_workers:int = 0):
    path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

    train_dataset = LFWDataset(path,train=True)
    test_dataset = LFWDataset(path,train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers
    )
    return train_loader,test_loader,train_dataset.num_positives,train_dataset.num_negatives



def make_casia_dataloaders(batch_size=256,shuffle=True,num_workers:int = 0):
    path = kagglehub.dataset_download("ntl0601/casia-webface")
    train_dataset = CasiaDataset(path,train=True)
    test_dataset = CasiaDataset(path,train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers
    )
    return train_loader,test_loader,train_dataset.num_positives,train_dataset.num_negatives

