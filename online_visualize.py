#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:02:55 2023

@author: kanchan
"""
import cv2
import torch
#import torchvision.models as models
import matplotlib.pyplot as plt
from model_demo_final import Model
from  preprocess import get_base_data
import opts
import numpy as np


# Create a function to load video features and process them
def process_video_features(video_features,model):
   
    model.eval()
    # Convert video_features to a PyTorch tensor
    data = get_base_data(video_features).unsqueeze(0)
#   
   

    # Forward pass through the model
    confidence_map, start, end, attention_weights  = model(data)
    attention_weights = attention_weights.transpose(1,2)
#    print("before attention_weights",attention_weights.shape)

    # Access the attention weights (specific to your model)

    # Normalize attention_weights if necessary
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())

    return attention_weights

# Load video features (replace with your actual video features)
def visualization(video_features,model):
     cap = cv2.VideoCapture("/activity_videos/v_0wwx4wnHv-U.mp4")
     while True:
        ret, frame = cap.read()
       
        if not ret:
            break  # Exit when there are no more frames
        with torch.no_grad():
            attention_weights = process_video_features(video_features,model)
            attention_weights = attention_weights.cpu().data.numpy()
#            print("attention_weights",attention_weights.shape)  #(1,100,256)
#        heatmap_colormap = plt.get_cmap('viridis')(attention_weights) * 255
        normalized_heatmap = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
      
        plt.figure(figsize=(10, 5))
        plt.imshow(attention_weights, cmap='YlGnBu', aspect='auto')
        plt.xlabel("Time Steps (Frames)")
        plt.ylabel("Attention Weights")
        plt.title("Temporal Attention in Video")
        plt.show()

        
    # Resize the heatmap to match the frame dimensions
        heatmap_resized = cv2.resize(normalized_heatmap, (frame.shape[1], frame.shape[0]))
      
         # Sum or aggregate the heatmap channels (each channel represents an attention head)
        combined_heatmap = np.sum(heatmap_resized, axis=-1)
#        print("111111",combined_heatmap.shape)  #[720,1280]
    # Normalize the combined heatmap to values between 0 and 1
        normalized_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min())
      
        # Apply a color map to the normalized heatmap
#        heatmap_colormap = plt.get_cmap('viridis')(normalized_heatmap) * 255
        heatmap_colormap = plt.get_cmap('viridis').reversed()(normalized_heatmap) * 255
        # Convert the heatmap colormap to a 3-channel RGB image
        heatmap_rgb = (heatmap_colormap[:, :, :3] * 255).astype(np.uint8)
     
        alpha = 0.5  # Adjust the transparency level
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_rgb, alpha, 0)
  
    # Display or save the frame with the overlay
    # Display the frame (you can replace this with saving the frames)
#    
        cv2.imshow('Frame with Overlay', overlay)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
   

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    model = Model(opt)
    model.cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/model_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    video_features = "/video_1.mp4"
    get_features = process_video_features(video_features,model) 
#    process_features = visualization(video_features,model)  #visualization
#  