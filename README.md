# What is Inpainting?

Inpainting originated as a traditional graphical problem, focusing on how to fill blank regions in an image using information from the surrounding areas, rendering the missing parts indistinguishable to the human eye. While humans may find this task relatively easy, computers face significant challenges in achieving satisfactory results. This difficulty arises due to the lack of a perfect solution and the complexities involved in utilizing the available information within the image to produce realistic outcomes.

## Inpainting Algorithms

Inpainting algorithms can generally be categorized into two main classes:

1. **Texture Synthesis:**
   - These algorithms focus on generating large image regions based on sample textures, leveraging the structural content of the provided images.

2. **Inpainting Techniques:**
   - These techniques are designed to fill in small image gaps, addressing specific areas of missing information.

In this project, our emphasis will be on the "Region Filling and Object Removal by Exemplar-Based Image Inpainting" proposed by A. Criminisi*, P. P´erez, and K. Toyama[1].

## Additional Research

Apart from the aforementioned techniques, we have also explored other approaches such as:

- **Scene Completion Using Millions of Photographs:**
  This approach entails leveraging a vast amount of data for scene completion. However, due to the extensive resource requirements, it may not be the most feasible option for our project.

- **Context Encoders: Feature Learning by Inpainting:**
  This paper, published in CVPR, introduces a methodology involving convolutional neural networks (CNNs) to learn high-level features in images. These features are then utilized to guide the generation of missing parts of images. The network structure proposed in the article consists of three parts: Encoder, Channel-wise fully connected layer, and Decoder. While this approach shows promise, its implementation requires a significant understanding of machine learning techniques, which we currently lack due to time constraints.
