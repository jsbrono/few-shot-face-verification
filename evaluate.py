import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np

def simple_evaluate_binary(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    total = 0
    total_correct = 0

    with torch.no_grad():
        for reference_data, query_data, labels, mask in test_loader:
            reference_data = reference_data.to(device)
            query_data = query_data.to(device)
            labels = labels.to(device).unsqueeze(-1)
            mask = mask.to(device).unsqueeze(-1)
            preds = model(reference_data, query_data, mask)
            
            predictions = (preds>0.5).float()
            total_correct += torch.sum(predictions==labels)
            total+=len(labels)


    print(f"Test acc: {total_correct}/{total}")
    return total_correct

def simple_evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    all_labels = []
    all_distances = []

    with torch.no_grad():
        for reference_data, query_data, labels, mask in test_loader:
            reference_data = reference_data.to(device)
            query_data = query_data.to(device)
            labels = labels.to(device).unsqueeze(-1)
            mask = mask.to(device).unsqueeze(-1)

            mean_ref_embeddings, query_embeddings = model(reference_data, query_data, mask)
            dist = F.pairwise_distance(mean_ref_embeddings.squeeze(), query_embeddings.squeeze())

            all_distances.append(dist.cpu())
            all_labels.append(labels.cpu())

    all_distances = torch.cat(all_distances).numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()

    thresholds = np.linspace(all_distances.min(), all_distances.max(), 1000)
    best_correct = 0
    best_thresh = 0.0
    for thresh in thresholds:
        preds = (all_distances < thresh).astype(int)
        correct = (preds == all_labels).sum()
        if correct > best_correct:
            best_correct = correct
            best_thresh = thresh

    print(f"Test set correct: {best_correct}/{len(all_labels)}")
    return best_correct, len(all_labels)


def simple_evaluate_nomask(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    all_labels = []
    all_distances = []

    with torch.no_grad():
        for reference_data, query_data, labels in test_loader:
            reference_data = reference_data.to(device)
            query_data = query_data.to(device)
            labels = labels.to(device).unsqueeze(-1)

            mean_ref_embeddings, query_embeddings = model(reference_data, query_data)
            dist = F.pairwise_distance(mean_ref_embeddings.squeeze(), query_embeddings.squeeze())

            all_distances.append(dist.cpu())
            all_labels.append(labels.cpu())

    all_distances = torch.cat(all_distances).numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()

    # Find best threshold
    thresholds = np.linspace(all_distances.min(), all_distances.max(), 1000)
    best_correct = 0
    best_thresh = 0.0
    for thresh in thresholds:
        preds = (all_distances < thresh).astype(int)
        correct = (preds == all_labels).sum()
        if correct > best_correct:
            best_correct = correct
            best_thresh = thresh

    print(f"Test set correct: {best_correct}/{len(all_labels)}")
    return best_correct, best_thresh



def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    all_labels = []
    all_distances = []
    all_query_images = []
    all_reference_images = []
    all_masks = []

    with torch.no_grad():
        for reference_data, query_data, labels, mask in test_loader:
            reference_data = reference_data.to(device)
            query_data = query_data.to(device)
            labels = labels.to(device).unsqueeze(-1)
            mask = mask.to(device).unsqueeze(-1)

            mean_ref_embeddings, query_embeddings = model(reference_data, query_data, mask)
            dist = F.pairwise_distance(mean_ref_embeddings.squeeze(), query_embeddings.squeeze())
            
            all_distances.append(dist.cpu())
            all_labels.append(labels.cpu())
            all_query_images.append(query_data.cpu())
            all_reference_images.append(reference_data.cpu())
            all_masks.append(mask.cpu())

    all_distances = torch.cat(all_distances).numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()
    all_query_images = torch.cat(all_query_images)
    all_reference_images = torch.cat(all_reference_images)
    all_masks = torch.cat(all_masks).squeeze() 

    # Find best threshold
    thresholds = np.linspace(all_distances.min(), all_distances.max(), 1000)
    best_acc = 0.0
    best_thresh = 0.0
    best_correct = 0.0
    for thresh in thresholds:
        preds = (all_distances < thresh).astype(int)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_correct = (preds==all_labels).sum()
            best_acc = acc
            best_thresh = thresh

    print(f"Best threshold: {best_thresh:.4f}, Accuracy: {best_acc:.4f} Total correct: {best_correct}/{len(all_labels)}")

    # Identify misclassified samples
    preds = (all_distances < best_thresh).astype(int)
    misclassified_mask = preds != all_labels
    misclassified_queries = all_query_images[misclassified_mask]
    misclassified_references = all_reference_images[misclassified_mask]
    misclassified_labels = all_labels[misclassified_mask]
    misclassified_preds = preds[misclassified_mask]
    misclassified_distances = all_distances[misclassified_mask]
    misclassified_masks = all_masks[misclassified_mask]

    # Plot each misclassified query with all its references
    for i in range(len(misclassified_queries)):
        query_img = misclassified_queries[i].permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
        ref_imgs = misclassified_references[i]  # shape: (num_refs, C, H, W)
        mask = misclassified_masks[i]           # shape: (num_refs,)

        valid_refs = ref_imgs[mask.squeeze().bool()]
        # num_refs = ref_imgs.shape[0]
        num_refs = valid_refs.shape[0]

        fig, axes = plt.subplots(1, num_refs + 1, figsize=(3*(num_refs+1), 3))
        
        # Plot reference images
        for j in range(num_refs):
            ref_img = valid_refs[j].permute(1, 2, 0).numpy()
            axes[j].imshow(ref_img)
            axes[j].axis('off')
            axes[j].set_title(f'Ref {j+1}')
        
        # Plot query image
        axes[-1].imshow(query_img)
        axes[-1].axis('off')
        axes[-1].set_title('Query')
        
        fig.suptitle(f'True: {misclassified_labels[i]}, Pred: {misclassified_preds[i]}, Distance: {misclassified_distances[i]:.4f}', fontsize=10)
        plt.savefig("misclassifications/"+str(i)+".png")

    return best_thresh, best_acc


def evaluate1(model:torch.nn.Module,test_loader:torch.utils.data.DataLoader,device:torch.device):
    
    model.eval()
    all_labels = []
    all_distances = []
    with torch.no_grad():
        for reference_data, query_data, labels, mask in test_loader:
            reference_data = reference_data.to(device)
            query_data = query_data.to(device)
            labels = labels.to(device).unsqueeze(-1)
            mask = mask.to(device).unsqueeze(-1)

            mean_ref_embeddings,query_embeddings = model(reference_data, query_data,mask)
            dist = F.pairwise_distance(mean_ref_embeddings.squeeze(), query_embeddings.squeeze())
            
            all_distances.append(dist.cpu())
            all_labels.append(labels.cpu())

    all_distances = torch.cat(all_distances).numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()

    thresholds = np.linspace(all_distances.min(), all_distances.max(), 1000)
    best_acc = 0.0
    best_thresh = 0.0

    for thresh in thresholds:
        preds = (all_distances < thresh).astype(int)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(f"Best threshold: {best_thresh:.4f}, Accuracy: {best_acc:.4f}")
    return best_thresh, best_acc