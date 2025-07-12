# Deep Learning - Traffic Density Classification

This notebook leverages a custom dataset collected from Singapore's Land Transport Authority (LTA) API to build and evaluate **deep learning models** for traffic density classification. The images are labeled across five traffic density levels and are captured under varying conditions, including different lighting and camera angles.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Data Splitting](#data-splitting)
- [Usage Instructions](#usage-instructions)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

This project is centered around developing a robust deep learning solution to classify traffic density based on images captured from multiple traffic cameras. The dataset includes images taken during both daytime and nighttime, ensuring that the models are well-equipped to handle diverse lighting and environmental conditions.

---

## Dataset Description

- **Source:** Singapore Land Transport Authority (LTA) API  
- **Image Count:** 4,054 images  
- **Traffic Density Categories:**  
  - **Empty**  
  - **Low**  
  - **Medium**  
  - **High**  
  - **Traffic Jam**  
- **Camera Selection:**  
  Out of 87 available traffic cameras, 20 were carefully selected based on:
  - Variability in traffic density  
  - Different lighting conditions  
  - Clear visibility without obstructions  
  - Suitable camera angles  

The dataset provides a well-rounded collection of images, making it ideal for training, validating, and testing deep learning models focused on traffic analysis.

---

## Data Splitting

The dataset has been divided into three subsets to ensure robust model evaluation:

- **Training Set:** 80% of the dataset  
- **Validation Set:** 10% of the dataset  
- **Testing Set:** 10% of the dataset

This split enables effective model training while ensuring that performance metrics are evaluated on unseen data.

---

## Usage Instructions

1. **Clone or Download the Notebook:**
   - Access the notebook via [Google Colab](https://colab.research.google.com/drive/1RSrHFGFSnMB2KW-m2ZJbmN1A9UmMoq0J#scrollTo=ji-bwSQzhucD).

2. **Set Up Environment:**
   - Ensure that you have the necessary libraries installed. The required dependencies are listed below.

3. **Run the Notebook:**
   - Follow the notebook sections sequentially to preprocess data, train the deep learning model, and evaluate its performance.
   - Modify parameters as needed to experiment with different model architectures or training configurations.
     
NB: 
I had to subscribe to Google Colab’s premium tier to get faster GPU access for the convolutional neural network to handle high-resolution image classification. On my 2024 MacBook Air with the M3 chip — powerful as it is — the process could take around 20 hours. By switching to a cloud-based GPU (in this case, the NVIDIA P100), runtime drops to about 30–45 minutes. Colab’s architecture supports both Python and R, and grants up to 89.6 GB of RAM, making it a crucial resource for memory-intensive tasks. If you need to run something that pushes beyond what a local CPU can handle, I highly recommend it! If you think we are friends, let me know and I can grant you access to my paid subscription.

---

## Dependencies

Ensure the following Python libraries are installed:
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV

You can install these dependencies via pip:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python
```

---

## Acknowledgements

- **Singapore Land Transport Authority (LTA):**  
- **Sudhanshu Rastogi:** 

---

## License

This project is licensed under the MIT License. Please refer to the [LICENSE](LICENSE) file for details.
