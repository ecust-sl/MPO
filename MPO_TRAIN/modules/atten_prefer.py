import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self, image_feature_dim, preference_vector_dim):
        super(FeatureFusion, self).__init__()
        self.image_feature_dim = image_feature_dim
        self.preference_vector_dim = preference_vector_dim
        self.projection = nn.Linear(preference_vector_dim, image_feature_dim)

    def forward(self, image_features, preference_vector):
        # ������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������ (batch_size, preference_vector_dim)
        batch_size = image_features.size(0)
        device = image_features.device
        dtype = image_features.dtype

        preference_vector = torch.tensor(preference_vector)
        preference_vector = preference_vector.to(device).type(dtype)
        preference_vector = preference_vector.expand(batch_size, -1)

        # ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
        projected_preference = self.projection(preference_vector)  # (batch_size, image_feature_dim)

        # ���������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
        attention_scores = torch.bmm(image_features.unsqueeze(1), projected_preference.unsqueeze(2)).squeeze()
        attention_weights = F.softmax(attention_scores, dim=-1)

        # ������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
        attention_weights = attention_weights.unsqueeze(-1)

        # ������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
        fused_features = image_features * attention_weights  # (batch_size, image_feature_dim)

        return fused_features
# class PreferenceFusion(nn.Module):
#     def __init__(self, input_dim, feature_dim, influence_factor=2):
#         super(PreferenceFusion, self).__init__()
#         self.fc_pref = nn.Linear(input_dim, feature_dim)
#         self.significance = nn.Linear(feature_dim, feature_dim)  # Significance scoring layer
#         self.influence_factor = influence_factor  # Amplification factor for preferences
#
#     def forward(self, decoder_outputs, preference_vector):
#         # Process and amplify preference vector
#         preference_vector = self.fc_pref(preference_vector.float())
#         amplified_preference = preference_vector * self.influence_factor
#
#         # Expand and apply significance scores
#         significance_scores = torch.sigmoid(self.significance(amplified_preference)).unsqueeze(1)
#         amplified_preference = amplified_preference.unsqueeze(1).expand_as(decoder_outputs)
#
#         # Dynamically adjust decoder outputs
#         adjusted_outputs = decoder_outputs + significance_scores * amplified_preference
#
#         return adjusted_outputs

class PreferenceFusion(nn.Module):
    def __init__(self, input_dim, feature_dim, influence_factor=40):
        super(PreferenceFusion, self).__init__()
        self.fc_pref = nn.Linear(input_dim, feature_dim)
        self.significance = nn.Linear(feature_dim, feature_dim)  # Significance scoring layer
        self.influence_factor = influence_factor  # Amplification factor for preferences

    def forward(self, decoder_outputs, preference_vector):
        # Process and amplify preference vector
        preference_vector = self.fc_pref(preference_vector.float())
        x1, x2 = preference_vector[0][0], preference_vector[0][1]  # Assuming preference_vector has size [1, feature_dim]
        # Apply influence factors to all elements of the preference vector
        fa1, fa2 = x1 * self.influence_factor, x2 * self.influence_factor
        preference_vector[:, 0] = fa1
        preference_vector[:, 1] = fa2

        amplified_preference = preference_vector

        # Expand and apply significance scores
        significance_scores = torch.sigmoid(self.significance(amplified_preference)).unsqueeze(1)
        amplified_preference = amplified_preference.unsqueeze(1).expand_as(decoder_outputs)

        # Dynamically adjust decoder outputs
        weight_factor = 0.5  # This can be adjusted based on how much you want to influence the decoder outputs
        adjusted_outputs = decoder_outputs + weight_factor * significance_scores * amplified_preference
        # adjusted_outputs = decoder_outputs + significance_scores * amplified_preference

        return adjusted_outputs

# class EnhancedDynamicChannelAttention(nn.Module):
#     def __init__(self, preference_dim, feature_channels):
#         super(EnhancedDynamicChannelAttention, self).__init__()
#         assert feature_channels % preference_dim == 0, "Feature channels must be divisible by the number of heads"
#         self.num_heads = preference_dim
#         self.head_dim = feature_channels // self.num_heads
#         self.attention_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(1, self.head_dim // 2),
#                 nn.Tanh(),
#                 nn.Dropout(0.5),
#                 nn.Linear(self.head_dim // 2, self.head_dim)
#             ) for _ in range(self.num_heads)
#         ])
#         self.residual_scale = 300

#     def forward(self, features, preference):
#         batch, step, seq = features.size()
#         expanded_preference = preference.unsqueeze(1).expand(batch, step, -1)
#         rougeWeights = float(preference[0][1])
#         # print(rougeWeights)

#         head_outputs = []
#         for i, head in enumerate(self.attention_heads):
#             # Ensure that preference slice is properly shaped for Linear layer
#             pref_slice = expanded_preference[:, :, i:i + 1].view(batch * step, 1)
#             weights = head(pref_slice)
#             weights = F.softmax(weights, dim=1).view(batch, step, self.head_dim)  # Reshape weights correctly

#             # Apply to the corresponding slice of features
#             feature_slice = features[:, :, i * self.head_dim:(i + 1) * self.head_dim]
#             # print(feature_slice.shape)
#             # print(weights.shape)
#             weighted_features = feature_slice * weights  # Ensure broadcasting
#             head_outputs.append(weighted_features)

#         # Concatenate outputs from all heads
#         weighted_features = torch.cat(head_outputs, dim=-1)
#         factor = self.residual_scale - rougeWeights * 100
#         weighted_features = factor * weighted_features + features

#         return weighted_features
class EnhancedDynamicChannelAttention(nn.Module):
    def __init__(self, preference_dim, feature_channels):
        super(EnhancedDynamicChannelAttention, self).__init__()
        assert feature_channels % preference_dim == 0, "Feature channels must be divisible by the number of heads"
        self.num_heads = preference_dim
        self.head_dim = feature_channels // self.num_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.head_dim // 2),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim // 2, self.head_dim)
            ) for _ in range(self.num_heads)
        ])
        self.query_layer = nn.Linear(1, self.head_dim)
        self.key_layer = nn.Linear(self.head_dim, self.head_dim)
        self.value_layer = nn.Linear(self.head_dim, self.head_dim)
        self.residual_scale = 700

    def forward(self, features, preference):
        # batch, step, seq = features.size()
        # expanded_preference = preference.unsqueeze(1).expand(batch, step, -1)
        # rougeWeights = float(preference[0][1])

        # head_outputs = []
        # for i, head in enumerate(self.attention_heads):
        #     pref_slice = expanded_preference[:, :, i:i + 1].view(batch * step, 1)
        #     weights = head(pref_slice)
        #     weights = F.softmax(weights, dim=1).view(batch, step, self.head_dim)  # Reshape weights correctly

        #     # Apply to the corresponding slice of features
        #     feature_slice = features[:, :, i * self.head_dim:(i + 1) * self.head_dim]
        #     # print(feature_slice.shape)
        #     # print(weights.shape)
        #     weighted_features = feature_slice * weights  # Ensure broadcasting
        #     head_outputs.append(weighted_features)

        # weighted_features = torch.cat(head_outputs, dim=-1)
        # factor = self.residual_scale - 200 * rougeWeights
        # weighted_features = factor * weighted_features + features
        batch, step, seq = features.size()
        expanded_preference = preference.unsqueeze(1).expand(batch, step, -1)

        head_outputs = []
        for i in range(self.num_heads):
            # Ensure that preference slice is properly shaped for Linear layer
            pref_slice = expanded_preference[:, :, i:i + 1].view(batch * step, 1)
            
            # Compute query from preference
            query = self.query_layer(pref_slice).view(batch, step, self.head_dim)
            
            # Compute key and value from features
            feature_slice = features[:, :, i * self.head_dim:(i + 1) * self.head_dim]
            key = self.key_layer(feature_slice)
            value = self.value_layer(feature_slice)
            
            # Compute attention weights
            weights = torch.bmm(query, key.transpose(1, 2))
            weights = F.softmax(weights, dim=-1)
            
            # Apply attention weights to value
            weighted_features = torch.bmm(weights, value)
            head_outputs.append(weighted_features)

        # Concatenate outputs from all heads
        weighted_features = torch.cat(head_outputs, dim=-1)
        factor = 1
        weighted_features = factor * weighted_features + features
        return weighted_features
        
class EnhancedDynamicChannelAttention_update(nn.Module):
    def __init__(self, preference_dim, feature_channels):
        super(EnhancedDynamicChannelAttention_update, self).__init__()
        assert feature_channels % preference_dim == 0, "Feature channels must be divisible by the number of heads"
        self.num_heads = preference_dim
        self.head_dim = feature_channels // self.num_heads

        # ������������������������������������
        self.attention_heads_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.head_dim // 2),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim // 2, self.head_dim)
            ) for _ in range(self.num_heads)
        ])

        self.attention_heads_2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.head_dim // 2),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim // 2, self.head_dim)
            ) for _ in range(self.num_heads)
        ])

        self.residual_scale = 800

    def forward(self, features, preference):
        batch, step, seq = features.size()
        expanded_preference = preference.unsqueeze(1).expand(batch, step, -1)
        rougeWeights = float(preference[0][1])

        head_outputs = []
        for i in range(self.num_heads):
            # ������������������������������������
            pref_slice_1 = expanded_preference[:, :, 0:1].view(batch * step, 1)
            weights_1 = self.attention_heads_1[i](pref_slice_1)
            weights_1 = F.softmax(weights_1, dim=1).view(batch, step, self.head_dim)

            # ������������������������������������
            pref_slice_2 = expanded_preference[:, :, 1:2].view(batch * step, 1)
            weights_2 = self.attention_heads_2[i](pref_slice_2)
            weights_2 = F.softmax(weights_2, dim=1).view(batch, step, self.head_dim)

            # ���������������������������
            combined_weights = (weights_1 + weights_2) / 2.0

            # ������������������������������
            feature_slice = features[:, :, i * self.head_dim:(i + 1) * self.head_dim]
            weighted_features = feature_slice * combined_weights  # Ensure broadcasting
            head_outputs.append(weighted_features)

        # ������������������������
        weighted_features = torch.cat(head_outputs, dim=-1)
        factor = self.residual_scale - rougeWeights * 400
        weighted_features = factor * weighted_features + features

        return weighted_features


