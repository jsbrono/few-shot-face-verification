import torch.nn as nn
import torch
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F


class EmbeddingCNN2D(nn.Module):
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
        super(EmbeddingCNN2D, self).__init__()

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
        #     nn.Flatten(),
        #     nn.Linear(128*4*4,  self._embedding_dim)
        # )

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.ann =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, self._embedding_dim),
        )




    def forward(self, reference_data: torch.Tensor, query_data: torch.Tensor,mask:torch.Tensor):
        
        # Assume reference_data is shape (batch,num_references,C,H,W)
        # query_data should be (batch,1,C,H,W,)
        B,R,C,H,W = reference_data.shape
        reference_data_stacked = reference_data.reshape(B*R,C,H,W)
        reference_embeddings = self.get_embedding(reference_data_stacked).reshape(B,R,-1)

        query_embedding = self.get_embedding(query_data)

        masked_reference_embeddings = reference_embeddings * mask
        mean_reference_embedding = torch.sum(masked_reference_embeddings,dim=1)/torch.sum(mask,dim=1)



        return mean_reference_embedding, query_embedding

    def get_embedding(self, data: torch.Tensor):
        """
        Get the embedding for a single data.

        Args:
            data (torch.Tensor):  data with shape (batch,   n   , features).

        Returns:
            torch.Tensor: Embedding for the input data.
        """
        # data = F.normalize(data,p=2,dim=1)?
        return self.ann(self.cnn(data))


class EmbeddingCNN3D_nomask(nn.Module):
    """
    CNN for embedding.

    This model consists of several convolutional layers followed by batch normalization and ReLU activations,
    designed to process 3D input data. The output from the convolutional layers is flattened.

    Attributes:
        cnn (nn.Sequential): Convolutional layers followed by batch normalization and ReLU activations.
    """

    def __init__(self):
        super(EmbeddingCNN3D_nomask, self).__init__()

        self._embedding_dim = 128
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.embed_linear =  nn.Sequential(
            nn.Linear(resnet.fc.in_features, self._embedding_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256, self._embedding_dim)
        )

    def forward(self, reference_data: torch.Tensor, query_data: torch.Tensor):
        
        reference_embedding = self.get_embedding(reference_data)
        query_embedding = self.get_embedding(query_data)


        return reference_embedding, query_embedding

    def get_embedding(self, data: torch.Tensor):
        """
        Get the embedding for a single data.

        Args:
            data (torch.Tensor):  data with shape (batch,   n   , features).

        Returns:
            torch.Tensor: Embedding for the input data.
        """
        data = self.cnn(data)
        data = torch.flatten(data,1)
        data = self.embed_linear(data)
        data = F.normalize(data,p=2,dim=1)
        return data
