# ece5831-2024-final-project

# Speech-to-Image Generation Using Fine-Tuned Latent Diffusion Models

## Table of Contents

1. [Motivation](#motivation)  
2. [Project Objectives](#project-objectives)  
3. [Significance](#significance)  
4. [Key Features](#key-features)  
5. [Dataset](#dataset)  
6. [Speech-to-Text Conversion](#speech-to-text-conversion)  
7. [Personalized Image Generation](#personalized-image-generation)  
8. [Model Architectures](#model-architectures)  
    - [Stable Diffusion Architecture](#stable-diffusion-architecture)  
    - [Fine-Tuning with DreamBooth](#fine-tuning-with-dreambooth)  
9. [Training Strategy](#training-strategy)  
10. [Performance Metrics](#performance-metrics)  
11. [Technologies Used](#technologies-used)  
12. [Contributors](#contributors)  
13. [Future Work](#future-work)  
14. [Project Attachments](#project-attachments)  

---

## Motivation  
With the growing demand for personalized content, this project bridges audio and visual modalities by generating images from speech. It has real-world applications in accessibility, content creation, and immersive environments like AR/VR. This innovation benefits visually impaired individuals and enhances creative workflows.

---

## Project Objectives  
- Develop a system that generates high-quality personalized images from speech inputs.  
- Seamlessly combine speech-to-text conversion and fine-tuned image generation models.  
- Optimize training efficiency and ensure high output quality.

---

## Significance  
The project advances multimodal AI by integrating speech recognition and image generation. It demonstrates how personalized models can enhance accessibility, creativity, and automation. This approach sets the groundwork for AI-driven tools in multiple industries.

---

## Key Features  
- Accurate speech-to-text conversion using OpenAI's Whisper model.  
- Fine-tuned Stable Diffusion with DreamBooth to generate personalized images.  
- Modular workflow ensuring efficiency, flexibility, and high-quality results.

---

## Dataset  
We created a custom dataset by capturing our own images to fine-tune the Stable Diffusion model. This small, targeted dataset ensures the generated images are highly personalized and contextually accurate.

---

## Speech-to-Text Conversion  
The OpenAI Whisper model accurately transcribes spoken input into text prompts. It handles multiple languages, various audio formats, and noisy environments effectively. These text prompts act as inputs for the image generation process.

---

## Personalized Image Generation  
Using Stable Diffusion v2, we fine-tuned the model with DreamBooth to generate personalized and context-specific images. By leveraging our dataset, the model produces results that are visually relevant and high-quality.

---

## Model Architectures  

### Stable Diffusion Architecture  
Stable Diffusion uses a latent diffusion process to generate images from text prompts. Its components include:  
- *U-Net*: Core neural network for noise prediction and removal.  
- *Variational Autoencoder (VAE)*: Compresses images into latent space.  
- *CLIP Text Encoder*: Translates text prompts into numeric representations.

### Fine-Tuning with DreamBooth  
DreamBooth enables efficient fine-tuning for personalized image generation. Key training parameters include prior preservation, batch size control, and memory-efficient optimizations like mixed precision.

---

## Training Strategy  
The model was fine-tuned with optimized configurations such as a low learning rate (1e-6), small batch sizes, and prior preservation techniques. Training involved iterative evaluations to monitor quality and ensure computational efficiency.

---

## Performance Metrics  
The performance was evaluated using metrics like relevance, visual fidelity, and quality of the generated images. Sample prompts were tested to validate the model’s ability to produce personalized results.

---

## Technologies Used  
- *Speech-to-Text*: OpenAI Whisper  
- *Image Generation*: Stable Diffusion v2  
- *Fine-Tuning Framework*: DreamBooth  
- *Optimizations*: Mixed precision training and memory-efficient optimizers

---

## Contributors  
- *Adil Qureshi*: Fine-tuning Stable Diffusion and model evaluation  
- *Shubham Jagtap*: Speech-to-text integration and testing
- *Jaskirat Sudan*: Dataset creation and image generation analysis  

---

## Future Work  
- Expand the dataset for better generalization and diversity.  
- Implement advanced metrics for cross-modal evaluation.  
- Integrate the model into augmented and virtual reality applications.  

---

## Project Attachments  
•⁠  ⁠*Presentation Link*: https://youtu.be/4edAK2OjUu4
•⁠  ⁠*Project Demo Link*: https://youtu.be/gDksvHqDzNs
•⁠  ⁠*Dataset Link*: https://drive.google.com/drive/folders/1DiNI8N2tjSfx6K9MFgu5IdIZVHb2Bi3Y?usp=sharing
•⁠  ⁠*Project Documents Link*: https://drive.google.com/drive/folders/1MUaLMb42silAoXdXlj5suvDwAEghulr5 
---
