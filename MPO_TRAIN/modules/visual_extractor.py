import torch
import torch.nn as nn
import torchvision.models as models

prefer_vector = None
def update_prefer_vector_image(prefer_vector_value):
    global prefer_vector
    prefer_vector = prefer_vector_value
class FusionModule(nn.Module):
    def __init__(self, d_model):
        super(FusionModule, self).__init__()
        self.fc = nn.Linear(2, d_model)  
        self.d_model = d_model

    def forward(self, images, prefer_vector):
        batch_size = images.size(0)  
        prefer_vector = torch.tensor(prefer_vector, dtype=torch.float32, device=images.device)


        prefer_vector_expanded = self.fc(prefer_vector)  # shape: [d_model]

       
        prefer_vector_expanded = prefer_vector_expanded.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # shape: [1, d_model, 1, 1]
        prefer_vector_expanded = prefer_vector_expanded.expand(batch_size, self.d_model, images.size(2), images.size(3))  # shape: [batch_size, d_model, 224, 224]

        
        images = images.to(prefer_vector_expanded.device)  
        fused_features = images + prefer_vector_expanded

        return fused_features
class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        # d_model = images.shape[1]
        # fusion_module = FusionModule(d_model).to(images.device)
        # images = fusion_module(images, prefer_vector).to(images.device)
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        # print('shape ====================', patch_feats.shape)
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
