from models.EmbeddingCNN import EmbeddingCNN3D, DifferentialEmbeddingCNN3D
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.utils import make_dynamic_signature_dataloaders, make_lfw_dataloaders, make_casia_dataloaders
from train import train, train_base
from evaluate import evaluate
from models.losses import EuclideanContrastiveLoss

if __name__ == "__main__":

    train_loader, test_loader, num_positives, num_negatives = make_casia_dataloaders(
        batch_size=128, shuffle=True, num_workers=4
    )

    # train_loader, test_loader, num_positives, num_negatives = make_lfw_dataloaders(
    #     batch_size=128, shuffle=True, num_workers=4
    # )

    # train_loader, test_loader = make_dynamic_signature_dataloaders(
    #     batch_size=64, shuffle=True,num_workers=4
    # )
    # num_positives = 1649
    # num_negatives = 762

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = EmbeddingCNN3D().to(device)
    model = DifferentialEmbeddingCNN3D().to(device)

    pos_weight = (num_positives + num_negatives) / (2 * num_positives)
    neg_weight = (num_positives + num_negatives) / (2 * num_negatives)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    # loss_fn = EuclideanContrastiveLoss(
    #     margin=1.0, pos_weight=pos_weight, neg_weight=neg_weight
    # )
    optimiser = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)

    num_epochs = 50

    train_base(model, train_loader, optimiser, loss_fn, num_epochs, device,test_loader,unimproved_epochs_max=30)

