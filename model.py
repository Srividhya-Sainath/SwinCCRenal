import torch
import torch.nn as nn
from timm import create_model
from utils import *

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: [bag_size, feature_dim]
        Returns: [feature_dim] (weighted average of features)
        """
        weights = self.attention(x)  # Shape: [bag_size, 1]
        weights = torch.softmax(weights, dim=0)  # Normalize weights across patches
        x = (weights * x).sum(dim=0)  # Weighted sum of features
        return x
    
class SwinTransformerFineTune(nn.Module):
    def __init__(self, num_classes, hidden_dim1, hidden_dim2, dropout_prob):
        super(SwinTransformerFineTune, self).__init__()
        
        self.swin = create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=0  # Removing the classification head
        )

        # Freezing the first 3 Stages
        for name, param in self.swin.named_parameters(): 
            if "layers.0" in name or "layers.1" in name or "layers.2" in name:
                param.requires_grad = False

        # Keep Stage 4 and classification head trainable
        for name, param in self.swin.named_parameters():
            if "layers.3" in name or "head" in name:  # Stage 4 and head
                param.requires_grad = True

        # Attention pooling
        self.attention_pooling = AttentionPooling(self.swin.num_features)

        # Adding a custom classification head for multi-label classification
        self.classifier = nn.Sequential(
            nn.Linear(self.swin.num_features, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim2, num_classes),
            nn.Sigmoid()
        )

        self.feature_norm = nn.BatchNorm1d(self.swin.num_features)

    def forward(self, bags):
        """
        Forward pass for the MIL model.
        Args:
            bags (List[torch.Tensor]): List of bags, each of shape [bag_size, 3, 224, 224].
        Returns:
            torch.Tensor: Predictions for each bag, shape [batch_size, num_classes].
        """
        bag_features = []
        for bag in bags:
            # Forward pass through Swin for each patch in the bag
            patches = self.swin(bag)  # Shape: [bag_size, feature_dim]
            patches = self.feature_norm(patches)
            # Instead of using mean pooling, we use attention pooling
            bag_feature = self.attention_pooling(patches)
            #bag_feature = patches.mean(dim=0)
            bag_features.append(bag_feature)
        
        bag_features = torch.stack(bag_features)  # Shape: [batch_size, feature_dim]
        output = self.classifier(bag_features)    # Shape: [batch_size, num_classes]
        return output

# if __name__ == "__main__":
#     num_classes = 3  # BAP1, PBRM1, SETD2
#     config_path = "/mnt/bulk-ganymede/vidhya/crick/SwinApproach/config.yaml"  # Path to the YAML config file
#     config = load_config(config_path)

#     model_config = config['MODEL']
#     num_classes = model_config['num_classes']
#     hidden_dim1 = model_config['hidden_dim1']
#     hidden_dim2 = model_config['hidden_dim2']
#     dropout_prob = model_config['dropout_prob']

#     model = SwinTransformerFineTune(num_classes, hidden_dim1, hidden_dim2, dropout_prob)

#     swin_trainable_params = count_trainable_params(model.swin)
#     classifier_trainable_params = count_trainable_params(model.classifier)
    
#     print(f"Trainable Parameters in Swin Transformer: {swin_trainable_params}")
#     print(f"Trainable Parameters in Classifier: {classifier_trainable_params}")
#     print(f"Total Trainable Parameters: {swin_trainable_params + classifier_trainable_params}")

#     dummy_input = torch.randn(2, 4, 3, 224, 224) # (batch_size, bag_size, channel, height, weight)
    
#     output = model(dummy_input)
#     print(f"Model Output: {output}")  # Outputs probabilities for each class