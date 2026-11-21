import torch.nn as nn
import torch
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F


class DifferentialEmbeddingCNN2D(nn.Module):
    """
    CNN followed by a fully connected layer for embedding extraction and prediction.

    This model consists of several convolutional layers followed by batch normalization and ReLU activations,
    designed to process 2D input data. The output from the convolutional layers is flattened and fed into
    a fully connected (ANN) module which outputs a single value representing a prediction (after sigmoid activation).

    Attributes:
        cnn (nn.Sequential): Convolutional layers followed by batch normalization and ReLU activations.
        ann (nn.Sequential): Fully connected layers for final prediction, followed by a sigmoid activation.

    """

    def __init__(self):
        super(DifferentialEmbeddingCNN2D, self).__init__()

        self._embeddings_dim = 1024
        self._mlp_input_dim = 100

        self.cnn = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),   
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(8, 16, 3, padding=1), 
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2), 

        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 32, 3, padding=1), 
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4,4)),

        nn.Flatten()
    )

        self.ann = nn.Sequential(
            nn.Linear(self._embeddings_dim, self._mlp_input_dim),
            nn.ReLU(),
            nn.Linear(self._mlp_input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, reference_data: torch.Tensor, unknown_data: torch.Tensor):
        """
        Forward pass of the Embedding CNN model.

        Args:
            reference_data (torch.Tensor): Reference data with shape (batch, ref_num, height, width).
            unknown_data (torch.Tensor): Unknown data with shape (batch, 1, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output of the model, mean reference differential embeddings,
            and mean unknown differential embeddings.

        The forward process involves the following steps:
        1. Process each reference data samples through the CNN to obtain embeddings.
        2. Process the unknown data sample through the same CNN to obtain an embedding.
        3. Compute reference differential embeddings by taking the absolute difference between the first reference embedding
           and the other reference embeddings.
        4. Calculate the mean of the differential embeddings for the reference data.
        5. Compute unknown differential embeddings by taking the absolute difference between the unknown embedding and
           each reference embedding.
        6. Calculate the mean of these differential embeddings for the unknown data.
        7. Concatenate the mean differential embeddings of the reference and unknown data.
        8. Pass the concatenated embeddings through a fully connected MLP.
        """
        ref_embeddings = []
        for i in range(reference_data.shape[1]):
            out = self.cnn(reference_data[:, i, ::].unsqueeze(1))
            ref_embeddings.append(out)
        #unknown_data = unknown_data.unsqueeze(1)
        unknown_embedding = self.cnn(unknown_data)

        ref_differential_embeddings = []
        for ref in ref_embeddings[1:]:
            ref_differential_embeddings.append(
                torch.abs(ref_embeddings[0] - ref)[:, None, :]
            )

        mean_reference_differential_embedding = torch.mean(
            torch.cat(ref_differential_embeddings, dim=1), dim=1
        )

        unk_differential_embeddings = []
        for ref in ref_embeddings[1:]:
            unk_differential_embeddings.append(
                torch.abs(unknown_embedding - ref)[:, None, :]
            )

        mean_unknown_differential_embedding = torch.mean(
            torch.cat(unk_differential_embeddings, dim=1), dim=1
        )

        x = torch.cat(
            [
                mean_reference_differential_embedding[:, None, :],
                mean_unknown_differential_embedding[:, None, :],
            ],
            dim=2,
        )
        x = self.ann(x)

        return (
            x,
            mean_reference_differential_embedding,
            mean_unknown_differential_embedding,
        )

    def get_embedding(self, data: torch.Tensor):
        """
        Get the embedding for a single data.

        Args:
            data (torch.Tensor):  data with shape (batch,   n   , features).

        Returns:
            torch.Tensor: Embedding for the input data.
        """
        return self.cnn(data.unsqueeze(1))

class DifferentialEmbeddingCNN3D(nn.Module):
    def __init__(self):
        super(DifferentialEmbeddingCNN3D, self).__init__()


        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.ann =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features*2, resnet.fc.in_features//2),
            nn.ReLU(),
            nn.Linear(resnet.fc.in_features//2,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    
    def forward_slow(self, reference_data: torch.Tensor, unknown_data: torch.Tensor, mask: torch.Tensor):

        B, R, C, H, W = reference_data.shape
        mask = mask.squeeze().bool()

        reference_data_stacked = reference_data.view(B * R, C, H, W)
        flat_mask = mask.view(-1).bool()
        valid_data = reference_data_stacked[flat_mask]

        valid_embeddings = self.get_embedding(valid_data) 

        reference_embeddings = torch.zeros(
            B * R, valid_embeddings.shape[-1],
            device=reference_data.device,
            dtype=valid_embeddings.dtype
        )
        reference_embeddings[flat_mask] = valid_embeddings
        reference_embeddings = reference_embeddings.view(B, R, -1) 

        query_embedding = self.get_embedding(unknown_data)  

        refs_i = reference_embeddings.unsqueeze(2)  
        refs_j = reference_embeddings.unsqueeze(1)  
        pairwise_diffs = torch.abs(refs_i - refs_j) 

        valid_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)) 
        triu_mask = torch.triu(torch.ones(R, R, device=mask.device, dtype=torch.bool), diagonal=1)
        full_pair_mask = valid_mask & triu_mask 

        # reference_pairwise_diffs = pairwise_diffs[full_pair_mask] 

        pairwise_sum = (pairwise_diffs * full_pair_mask.unsqueeze(-1)).sum(dim=(1, 2))
        pairwise_count = full_pair_mask.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)  
        mean_reference_pairwise = pairwise_sum / pairwise_count 

        query_diffs = torch.abs(query_embedding.unsqueeze(1) - reference_embeddings) 
        query_diffs = query_diffs * mask.unsqueeze(-1)  

        query_sum = query_diffs.sum(dim=1)  
        query_count = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mean_query_diffs = query_sum / query_count

        #testing
        # mean_query_diffs = query_embedding

        x = torch.cat([mean_reference_pairwise, mean_query_diffs], dim=1) 
        x = self.ann(x)

        return x

    def forward(self, reference_data: torch.Tensor, unknown_data: torch.Tensor, mask: torch.Tensor):

        B, R, C, H, W = reference_data.shape

        reference_data_stacked = reference_data.view(B * R, C, H, W)
        flat_mask = mask.view(-1).bool()
        valid_data = reference_data_stacked[flat_mask]

        valid_embeddings = self.get_embedding(valid_data)  

        reference_embeddings = torch.zeros(
            B * R, valid_embeddings.shape[-1],
            device=reference_data.device,
            dtype=valid_embeddings.dtype
        )
        reference_embeddings[flat_mask] = valid_embeddings
        reference_embeddings = reference_embeddings.view(B, R, -1) 

        query_embedding = self.get_embedding(unknown_data) 

        valid_refs_mask = mask.bool()
        anchor = reference_embeddings[:, 0:1, :] 
        other_refs = reference_embeddings[:, 1:, :]  
        other_refs_mask = valid_refs_mask[:, 1:]  

        reference_diff = torch.abs(anchor - other_refs)  
        reference_diff_masked = reference_diff * other_refs_mask

        num_valid_refs = other_refs_mask.sum(dim=1).clamp(min=1)

        mean_reference_diff = reference_diff_masked.sum(dim=1) / num_valid_refs  

        query_diff = torch.abs(query_embedding.unsqueeze(1) - other_refs) 
        query_diff_masked = query_diff * other_refs_mask 

        mean_query_diff = query_diff_masked.sum(dim=1) / num_valid_refs  

        x = torch.cat([mean_reference_diff, mean_query_diff], dim=1) 
        x = self.ann(x)
        return x

    def get_embedding(self, data: torch.Tensor):
        return F.normalize(torch.flatten(self.cnn(data),start_dim=1),dim=-1)
    
    def forward_old(self, reference_data: torch.Tensor, unknown_data: torch.Tensor, mask):

        B,R,C,H,W = reference_data.shape

        # reference_data_stacked = reference_data.view(B*R, C, H, W)
        # reference_embeddings_stacked = self.get_embedding(reference_data_stacked)
        # reference_embeddings = reference_embeddings_stacked.view(B,R,-1)
        reference_data_stacked = reference_data.view(B*R, C, H, W)
        flat_mask = mask.view(-1)
        valid_data = reference_data_stacked[flat_mask.bool()]  

        valid_embeddings = self.get_embedding(valid_data)  

        reference_embeddings = torch.zeros(B*R, valid_embeddings.shape[-1], device=reference_data.device,dtype=valid_embeddings.dtype)
        reference_embeddings[flat_mask.bool()] = valid_embeddings
        reference_embeddings = reference_embeddings.view(B, R, -1)
            
        query_embedding = self.get_embedding(unknown_data)
        
        reference_differential_embeddings = torch.abs(reference_embeddings[:,0:1,:] - reference_embeddings[:,1:,:])

        mean_reference_differential_embedding = torch.mean(reference_differential_embeddings,dim=1)

        query_differential_embeddings = torch.abs(query_embedding.unsqueeze(1) - reference_embeddings[:,1:,:])

        mean_query_differential_embedding = torch.mean(query_differential_embeddings,dim=1)

        x = torch.cat([mean_reference_differential_embedding,mean_query_differential_embedding],dim=1)

        x = self.ann(x)

        return x

class EmbeddingCNN3D(nn.Module):
    """
    CNN for embedding.

    This model consists of several convolutional layers followed by batch normalization and ReLU activations,
    designed to process 3D input data. The output from the convolutional layers is flattened.

    Attributes:
        cnn (nn.Sequential): Convolutional layers followed by batch normalization and ReLU activations.
    """

    def __init__(self):
        super(EmbeddingCNN3D, self).__init__()

        self._embedding_dim = 128

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((4,4)), 
        # )
        # self.ann = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128*4*4, self._embedding_dim),
        # )

        # resnet is usable but above works just as well on signature dataset, but training time takes longer

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.ann =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, self._embedding_dim),
        )

    def forward(self, reference_data, query_data, mask:torch.Tensor):
        B, R, C, H, W = reference_data.shape
        
        reference_data_stacked = reference_data.view(B*R, C, H, W)
        flat_mask = mask.view(-1)
        valid_data = reference_data_stacked[flat_mask.bool()]  

        valid_embeddings = self.get_embedding(valid_data)  

        reference_embeddings = torch.zeros(B*R, self._embedding_dim, device=reference_data.device,dtype=valid_embeddings.dtype)
        reference_embeddings[flat_mask.bool()] = valid_embeddings
        reference_embeddings = reference_embeddings.view(B, R, -1)

        query_embedding = self.get_embedding(query_data)

        masked_reference_embeddings = reference_embeddings * mask
        mean_reference_embedding = torch.sum(masked_reference_embeddings, dim=1) / torch.sum(mask, dim=1)

        return mean_reference_embedding, query_embedding

    def get_embedding(self, data: torch.Tensor):
        """
        Get the embedding for a single data.

        Args:
            data (torch.Tensor):  data with shape (batch,   n   , features).

        Returns:
            torch.Tensor: Embedding for the input data.
        """
        return F.normalize(self.ann(self.cnn(data)),p=2,dim=1)