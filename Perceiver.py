#!/usr/bin/env python
# coding: utf-8

# In[25]:


import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np

import math


# In[26]:


torch.cuda.is_available()


# ## Attention Scoring (Scaled Dot Product)

# In[27]:


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, dropout):
        
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values):
        
        weights = torch.bmm(queries, torch.transpose(keys, 1, 2)) / math.sqrt(queries.shape[-1])
        
        self.attn_weights = F.softmax(weights, dim = -1)
        
        scores = torch.bmm(self.dropout(self.attn_weights), values)
        
        return scores    


# ## Normalization

# In[28]:


class Normalize(nn.Module):
    
    def __init__(self, norm_shape):
        
        super().__init__()
        
        self.norm = nn.LayerNorm(norm_shape)
        
    def forward(self, inputs):
        
        return self.norm(inputs)


# ## Residual Connection + Normalization

# In[29]:


class AddNorm(nn.Module):
    
    def __init__(self, norm_shape):
        
        super().__init__()
        
        self.norm = nn.LayerNorm(norm_shape)
        
    def forward(self, inputs, outputs):
        
        return self.norm(inputs + outputs)


# ## Initialize Latent Array

# In[30]:


class LatentArray(nn.Module):
    
    def __init__(self, latent_size, latent_dim):
        
        super().__init__()
        
        self.latent_size = latent_size
        
        self.latent_dim = latent_dim
        
    def forward(self, inputs):
        
        latent_array = nn.Parameter(torch.rand(self.latent_size, self.latent_dim))
        
        torch.nn.init.trunc_normal_(latent_array, mean = 0.0, std = 0.02, a = -2.0, b = 2.0)
        
        batch_latent_array = latent_array.repeat(inputs.shape[0], 1, 1)
        
        return batch_latent_array   


# ## Dense Multi-layer Perceptron Block

# In[31]:


class DenseBlock(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, bias = False):
        
        super().__init__()
        
        self.norm = nn.LayerNorm(input_dim)
        
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias = bias)
        
        self.layer2 = nn.Linear(hidden_dim, input_dim, bias = bias)
        
    def forward(self, inputs):
        
        # Layer Normalize the inputs
        
        outputs = self.norm(inputs)
        
        # Pass through the first linear layer & activate with GELU
        
        first_outputs = F.gelu(self.layer1(outputs))
        
        # Pass through the final linear layer
        
        final_outputs = self.layer2(first_outputs)
        
        return final_outputs


# ## Fully Connected FeedForward Neural Network Layer

# In[32]:


class FFNLayer(nn.Module):
    
    def __init__(self, embedded_dim, hidden_size, output_size):
        
        super().__init__()
        
        self.layer1 = nn.Linear(embedded_dim, hidden_size)
        
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        
        first_outputs = F.relu(self.layer1(inputs))
        
        final_outputs = self.layer2(first_outputs)
        
        return final_outputs


# ## Multi Heads Attention (for Self-Attention)

# In[33]:


class MultiHeadAttention(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, num_heads, dropout, bias = False):
        
        super().__init__()
        
        self.num_heads = num_heads
        
        self.attention = ScaledDotProductAttention(dropout)
        
        self.norm = Normalize(latent_dim)
        
        self.addnorm = AddNorm(latent_dim)
        
        self.queries_weights = nn.Linear(latent_dim, latent_dim, bias = bias)
        
        self.keys_weights = nn.Linear(latent_dim, latent_dim, bias = bias)
        
        self.values_weights = nn.Linear(latent_dim, latent_dim, bias = bias)
        
        self.heads_weights = nn.Linear(latent_dim, latent_dim, bias = bias)
        
        self.dense = DenseBlock(latent_dim, hidden_dim)
        
    def reshape_multi_heads(self, inputs):
        
        # batch size, number of queries/keys/values, number of heads, dimensions / number of heads
        
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], self.num_heads, -1)
        
        # batch size, number of heads, number of queries/keys/values, dimensions / number of heads
        
        inputs = inputs.permute(0, 2, 1, 3)
        
        # batch size x number of heads, number of queries/keys/values, dimensions / number of heads

        inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3])
        
        return inputs
    
    def reshape_output(self, outputs):
        
        # batch size, number of heads, number of queries/keys/values, dimensions / number of heads
        
        outputs = outputs.reshape(-1, self.num_heads, outputs.shape[1], outputs.shape[2])
        
        # batch size, number of queries/keys/values, number of heads, dimensions / number of heads

        outputs = outputs.permute(0, 2, 1, 3)
        
        # batch size, number of queries/keys/values, dimensions
        
        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], -1)
        
        return outputs
        
    def forward(self, queries, keys, values):
        
        # Layer Normalize inputs
        
        norm_queries = self.norm(queries)
        
        norm_keys = self.norm(keys)
        
        norm_values = self.norm(values)
        
        # Re-shape QKV into multiple heads and attach learnable parameters
        
        new_queries = self.reshape_multi_heads(self.queries_weights(norm_queries))
        
        new_keys = self.reshape_multi_heads(self.keys_weights(norm_keys))
        
        new_values = self.reshape_multi_heads(self.values_weights(norm_values))
        
        # Perform attention scoring method
        
        outputs = self.attention(new_queries, new_keys, new_values)
        
        # Re-shape the outputs into their original shape (same with keys & values)

        outputs = self.reshape_output(outputs)
        
        # Attach learnable parameters to heads
        
        outputs = self.heads_weights(outputs)
        
        # Residual Connection & Normalization
        
        outputs = self.addnorm(queries, outputs)
        
        # Pass through Dense Block
        
        final_outputs = self.dense(outputs)
        
        return final_outputs


# ## Cross Attention

# In[34]:


class CrossAttention(nn.Module):
    
    def __init__(self, embedded_dim, hidden_dim, latent_dim, dropout, bias = False):
        
        super().__init__()
        
        self.attention = ScaledDotProductAttention(dropout)
        
        self.latent_norm = Normalize(latent_dim)
        
        self.norm = Normalize(embedded_dim)
        
        self.addnorm = AddNorm(embedded_dim)
        
        self.latent_dim = latent_dim
        
        self.keys_weights = nn.Linear(embedded_dim, embedded_dim, bias = bias)
        
        self.values_weights = nn.Linear(embedded_dim, embedded_dim, bias = bias)
        
        self.in_latent_linear = nn.Linear(latent_dim, embedded_dim, bias = bias)
        
        self.out_latent_linear = nn.Linear(embedded_dim, latent_dim, bias = bias)
        
        self.dense = DenseBlock(latent_dim, hidden_dim)
        
    def forward(self, latent_array, keys, values):
        
        # Layer Normalize inputs
        
        norm_latent_array = self.latent_norm(latent_array)
        
        # Pass latent array to linear layer so its dimension is the same with keys & values
        
        latent_array_1 = self.in_latent_linear(norm_latent_array)
        
        # Normalize and attach learnable parameters to keys
        
        keys = self.norm(keys)
        
        keys = self.keys_weights(keys)
        
        # Normalize and attach learnable parameters to values
        
        values = self.norm(values)
        
        values = self.values_weights(values)
        
        # Perform cross attention for latent array with keys & values
        
        cross_outputs = self.attention(latent_array_1, keys, values)
        
        # Pass the outputs to a linear layer
        
        outputs = self.out_latent_linear(cross_outputs)
        
        # Return Residual Connection
        
        res_outputs = outputs + latent_array
        
        # Pass through Dense Block
        
        final_outputs = self.dense(res_outputs)
        
        return final_outputs


# ## Perceiver Block

# In[35]:


class PerceiverBlock(nn.Module):
    
    def __init__(self, num_heads, embedded_dim, hidden_dim, latent_dim, 
                 self_attention_modules, cross_attention_modules, dropout):
        
        super().__init__()
        
        # self.cross_attention = CrossAttention(embedded_dim, dropout)
        
        # self.self_attention = MultiHeadAttention(embedded_dim, num_heads, dropout)
        
        self.addnorm = AddNorm(embedded_dim)
        
        self.linear_layer = nn.Linear(embedded_dim, embedded_dim)
        
        self.latent_linear = nn.Linear(latent_dim, embedded_dim)
        
        self.ffn = FFNLayer(latent_dim, hidden_dim, latent_dim)
        
        self.cross_modules = nn.Sequential()
        
        self.self_modules = nn.Sequential()
        
        for i in range(cross_attention_modules):
            self.cross_modules.add_module("cross attention"+str(i), CrossAttention(embedded_dim, hidden_dim, 
                                                                                   latent_dim, dropout))
            
        for i in range(self_attention_modules):
            self.self_modules.add_module("self attention"+str(i), MultiHeadAttention(latent_dim, hidden_dim,
                                                                                     num_heads, dropout))
        
    def forward(self, latent_inputs, inputs):
        
        for i, cross_attention in enumerate(self.cross_modules):
            
            latent_inputs = cross_attention(latent_inputs, inputs, inputs)
            
        for i, self_attention in enumerate(self.self_modules):
            
            latent_inputs = self_attention(latent_inputs, latent_inputs, latent_inputs)
              
        return latent_inputs


# ## Perceiver

# In[36]:


class Perceiver(nn.Module):
    
    def __init__(self, embedded_dim, hidden_dim, num_heads, num_blocks, latent_size, latent_dim, 
                 self_attention_modules, cross_attention_modules, dropout):
        
        super().__init__()
        
        self.softmax = nn.Softmax(dim=1)
        
        self.blocks = nn.Sequential()
        
        self.ffn = FFNLayer(latent_dim, hidden_dim, 10)
        
        self.latent = LatentArray(latent_size, latent_dim)
        
        self.norm = nn.LayerNorm(latent_dim)
        
        self.linear = nn.Linear(latent_dim, latent_dim)
        
        for i in range(num_blocks):
            
            self.blocks.add_module("block"+str(i), PerceiverBlock(num_heads, embedded_dim, hidden_dim, latent_dim,
                                                                  self_attention_modules, cross_attention_modules, dropout))
        
    def forward(self, inputs):
        
        # Initialize Latent Array
        
        latent_outputs = self.latent(inputs).to("cuda:0")
        
        # Running the Perceiver Blocks sequentially
        
        for i, perceiver_block in enumerate(self.blocks):
            
            latent_outputs = perceiver_block(latent_outputs, inputs)
        
        outputs = latent_outputs.mean(dim = 1)
        
        outputs_ffn = self.ffn(outputs)
        
        outputs_final = self.softmax(outputs_ffn)
              
        return outputs_final


# In[37]:


# Pre-processing images

image_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5), (0.5))])


# In[38]:


# Downloading training and testing datasets

train_data = torchvision.datasets.MNIST(root = './mnist_data', train = True,
                                       download = False, transform = image_transforms)

test_data = torchvision.datasets.MNIST(root = './mnist_data', train = False,
                                       download = False, transform = image_transforms)


# In[39]:


# Loading training and testing datasets

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)


# In[40]:


image_iter = iter(train_dataloader)

images, labels = next(image_iter)


# In[41]:


images.shape


# In[42]:


labels.shape


# In[43]:


model = Perceiver(embedded_dim = 28, num_heads = 4, hidden_dim = 32, num_blocks = 6, latent_size = 7, latent_dim = 36,
                  self_attention_modules = 1, cross_attention_modules = 2, dropout = 0.2)


# In[44]:


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, amsgrad = True)

epochs = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[45]:


model = model.to(device)


# In[46]:


n_total_steps = len(train_dataloader)


# In[47]:


import time

start_time = time.time()


# In[48]:


for epoch in range(epochs):
    
    for i, (images, labels) in enumerate(train_dataloader):
        
        images = torch.squeeze(images, 1)
        
        # labels = torch.unsqueeze(labels, 1)
        
        images = images.to(device)
        
        labels = labels.to(device)
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        if (i+1) % 200 == 0:
            print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.3f}')


# In[49]:


print("--- %s seconds ---" % (time.time() - start_time))


# In[50]:


# List of labels

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


# In[51]:


# Evaluate the trained model performance

with torch.no_grad():
    
    batch_size = 16
    
    num_correct_preds = 0
    
    num_total = len(test_data)
    
    num_correct_per_label = [0] * len(classes)
    
    num_total_per_label = [0] * len(classes)
    
    for images, labels in test_dataloader:
        
        images = torch.squeeze(images, 1)
        
        images = images.to(device)
        
        labels = labels.to(device)
        
        outputs = model(images)
        
        # Return the value with the highest probability score
        
        _, pred_values = torch.max(outputs, 1)
        
        num_correct_preds += (pred_values == labels).sum().item()        
    
        for i in range(batch_size):
            
            label = labels[i]
            
            pred_val = pred_values[i]
            
            num_total_per_label[label] += 1 
            
            if label == pred_val:
                
                num_correct_per_label[label] += 1
                
    # Calculate Overall Accuracy
    
    overall_accuracy = 100.0 * num_correct_preds / num_total
    
    print(f'Overall accuracy of the model: {overall_accuracy} %')
    
    # Calculate Accuracy per Label
    
    for i in range(len(classes)):
        
        accuracy_per_label = 100.0 * num_correct_per_label[i] / num_total_per_label[i]
        
        print(f'Accuracy of Label {classes[i]} : {accuracy_per_label} %')

