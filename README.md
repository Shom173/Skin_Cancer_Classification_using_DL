# Skin_Cancer_Classification_using_DL

Here's a **README.md** file for your GitHub repository:  

---

# **Skin Cancer Classification Using CNN and VGG19**  

## **Overview**  
This project focuses on detecting and classifying skin cancer using **Convolutional Neural Networks (CNN)** and **VGG19**. By leveraging deep learning techniques, the model aims to assist in early diagnosis and improve classification accuracy.  

## **Dataset**  
- The dataset consists of labeled images of skin lesions.  
- Images are preprocessed with resizing, normalization, and augmentation to improve model performance.  

## **Methodology**  
1. **Data Preprocessing**: Image resizing, normalization, and augmentation.  
2. **Model Development**:  
   - **CNN Model**: A custom-built CNN with multiple convolutional layers.  
   - **VGG19 Model**: A pre-trained VGG19 model fine-tuned for classification.  
3. **Training & Optimization**:  
   - **Loss Function**: Categorical Cross-Entropy  
   - **Optimizer**: Adam  
   - **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score  
4. **Performance Analysis**:  
   - VGG19 outperforms CNN due to better feature extraction.  
   - Class imbalance addressed using re-weighted loss.  

## **Results**  
- **VGG19 achieved higher accuracy** and better generalization.  
- CNN performed well but was slightly less effective than VGG19.  
- **Future improvements**: Balancing dataset, hyperparameter tuning, and additional augmentation.  

## **Installation & Usage**  
### **Requirements**  
- Python 3.x  
- TensorFlow/Keras  
- OpenCV, NumPy, Pandas, Matplotlib  
