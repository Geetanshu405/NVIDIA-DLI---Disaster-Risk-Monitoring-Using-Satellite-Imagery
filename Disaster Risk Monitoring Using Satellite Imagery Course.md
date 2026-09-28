# Disaster Risk Monitoring Using Satellite Imagery
 Assessment — Answer Key

A concise study guide and answer key covering **Computer Vision, Sentinel-1 SAR data, machine learning workflows, NVIDIA TAO Toolkit, transfer learning, data augmentation, NGC, model evaluation, and flood mapping**.

> **Note:** This repository is intended as a learning/reference resource. The explanations are written to explain *why* an answer is correct rather than simply providing the answer.

---

## Table of Contents

- [Q1 — Computer Vision Tasks](#q1--computer-vision-tasks)
- [Q2 — Sentinel-1 Satellite Data](#q2--sentinel-1-satellite-data)
- [Q3 — Machine Learning Workflow](#q3--machine-learning-workflow)
- [Q4 — TAO Toolkit](#q4--tao-toolkit)
- [Q5 — Transfer Learning](#q5--transfer-learning)
- [Q6 — Data Augmentation](#q6--data-augmentation)
- [Q7 — NVIDIA NGC](#q7--nvidia-ngc)
- [Q8 — Dataset Splitting](#q8--dataset-splitting)
- [Q9 — Model Selection](#q9--model-selection)
- [Q10 — Flood Mapping](#q10--flood-mapping)
- [Quick Answer Key](#quick-answer-key)

---

# Q1 — Computer Vision Tasks

### Question

**Which of the following is a Computer Vision task for drawing insights from videos?**

### Options

- [ ] Classification to determine which objects are in an image.
- [ ] Object Detection combines classification and localization to determine what and where objects are in an image.
- [ ] Segmentation to provide pixel-wise masks generated for each object in the image.
- [x] **All of the above.**

### Correct Answer

**All of the above.**

### Explanation

Computer Vision isn't limited to one particular type of task. Classification, object detection, and segmentation are all commonly used to extract information from visual data.

- **Classification** identifies what is present in an image.
- **Object detection** identifies objects and also determines where they are, usually using bounding boxes.
- **Segmentation** goes a step further by identifying the pixels belonging to different objects or regions.

When working with video, these techniques can be applied to individual frames to track and understand what is happening over time.

---

# Q2 — Sentinel-1 Satellite Data

### Question

**What kind of data do the Sentinel-1 satellites collect?**  
*Choose all that apply.*

### Options

- [ ] High resolution RGB images.
- [x] **C-band synthetic aperture radar data.**
- [x] **Reliable data regardless of weather, light, or cloud condition.**
- [x] **Signals from Horizontal (H) and Vertical (V) dual polarimetric channels.**

### Correct Answers

**C-band synthetic aperture radar data**

**Reliable data regardless of weather, light, or cloud condition**

**Signals from Horizontal (H) and Vertical (V) dual polarimetric channels**

### Explanation

Sentinel-1 is based on **Synthetic Aperture Radar (SAR)** rather than conventional optical imaging. It operates in the **C-band** and can collect data during both day and night.

One of the major advantages of SAR is that it isn't dependent on sunlight and can acquire useful observations through cloud cover and many weather conditions. This makes Sentinel-1 particularly valuable for applications such as **flood monitoring, land-use mapping, agriculture, and disaster response**.

Sentinel-1 also uses different radar polarizations, which provide additional information about the characteristics of surfaces being observed.

---

# Q3 — Machine Learning Workflow

### Question

**Please choose the correct machine learning workflow.**

### Options

- [x] **Prepare Data → Train → Evaluate → Optimize → Evaluate → Export → Deploy**
- [ ] Prepare Data → Optimize → Train → Export → Deploy → Evaluate
- [ ] Prepare Data → Train → Optimize → Export → Deploy
- [ ] Prepare Data → Train → Evaluate → Export → Deploy → Optimize

### Correct Answer

**Prepare Data → Train → Evaluate → Optimize → Evaluate → Export → Deploy**

### Explanation

A machine learning project normally follows an iterative workflow.

First, the data needs to be prepared and made suitable for training. The model is then trained and evaluated to understand how well it performs. Based on the evaluation, the model can be optimized through changes to the model, training configuration, or other parameters.

After optimization, the model should be evaluated again to verify that the changes actually improved its performance. Only after this stage is the model ready to be exported and deployed.

The important part is that **evaluation happens both before and after optimization**. Optimization without evaluation makes it difficult to know whether the changes actually helped.

---

# Q4 — TAO Toolkit

### Question

**Which of the following is true about the TAO Toolkit?**  
*Choose all that apply.*

### Options

- [x] **A low-coding, AI training toolkit that makes it easy to start building custom AI models.**
- [x] **Uses transfer learning to reduce cost associated with building AI models.**
- [x] **Offers purpose-built pretrained models that are production ready.**
- [x] **Utilizes a CLI to perform tasks that are configured with spec files.**

### Correct Answers

**All of the above.**

### Explanation

The **NVIDIA TAO Toolkit** is designed to make custom AI model development easier without requiring developers to build and train everything from scratch.

A major part of TAO's approach is **transfer learning**. Instead of starting with a randomly initialized model, developers can begin with an existing pretrained model and adapt it to a specific use case.

TAO also provides pretrained models and uses a **command-line interface (CLI)** where training and other tasks can be configured using specification files. This makes experiments more reproducible and provides a structured way to configure the training process.

---

# Q5 — Transfer Learning

### Question

**Which of the following is false about transfer learning?**

### Options

- [ ] It's the process of transferring learned features from one application to another.
- [ ] Reduces data scarcity issues and time to build a model.
- [x] **Models trained from transfer learning are not as accurate as training from scratch.**
- [ ] It leverages the feature extraction layers from a pretrained model.

### Correct Answer

**Models trained from transfer learning are not as accurate as training from scratch.**

### Explanation

This statement is **false**.

Transfer learning uses knowledge learned from an existing model and applies it to a new but related task. Instead of learning every feature from the beginning, the new model can reuse useful representations learned during pretraining.

This can significantly reduce the amount of training data and compute required. Depending on the task and dataset, transfer learning can also achieve very strong accuracy and can sometimes outperform training a model entirely from scratch.

The actual performance depends on factors such as the similarity between the original and target tasks, dataset quality, model architecture, and fine-tuning strategy.

---

# Q6 — Data Augmentation

### Question

**Which of the following is false about data augmentation?**

### Options

- [ ] Data augmentation artificially increases the size of training data to achieve accurate results.
- [ ] Data augmentation includes geometric deformation, color transforms, and noise addition.
- [ ] Data augmentation helps produce models that are more robust in their predictions and less prone to overfitting.
- [x] **Data augmentation improves model training time by reducing the number of samples required for training.**

### Correct Answer

**Data augmentation improves model training time by reducing the number of samples required for training.**

### Explanation

This is the **false** statement.

Data augmentation creates modified versions of existing training examples. Common techniques include **rotation, cropping, flipping, scaling, color changes, geometric transformations, and adding noise**.

The goal is to expose the model to more variation during training. This can help the model generalize better to data it hasn't seen before and reduce overfitting.

However, augmentation does not inherently make training faster. In fact, generating additional variations can add computational work during training.

---

# Q7 — NVIDIA NGC

### Question

**What is the role of NGC?**  
*Choose all that apply.*

### Options

- [x] **It is a catalog of GPU-optimized software for AI practitioners to develop their AI solutions.**
- [x] **It provides access to various AI services for model training, deployment, and monitoring.**
- [x] **It provides a private registry for securely accessing and managing proprietary AI software.**
- [ ] It forces developers to open-source their AI solutions.

### Correct Answers

**1, 2, and 3**

### Explanation

**NVIDIA NGC** provides a collection of GPU-optimized resources designed to help developers and organizations build and deploy AI applications.

It includes resources such as **containers, pretrained models, software packages, and other optimized AI resources**.

NGC can also be used within organizations to securely manage and distribute proprietary AI software and other resources.

The final option is incorrect because NGC does **not** require developers to open-source their AI solutions. It can be used for both open and proprietary development workflows.

---

# Q8 — Training, Validation, and Testing Sets

### Question

**Why should the data set be split into training, validation, and testing sets for supervised machine learning?**

### Options

- [ ] It is better to split the data set into subsets if it is too large.
- [x] **To prevent the model from overfitting and to accurately evaluate the model.**
- [ ] We put the positive classes into the training set so we can build a model that can label negative classes in the test set.
- [ ] We put outlier data in the test set to see how the model will perform on them.

### Correct Answer

**To prevent the model from overfitting and to accurately evaluate the model.**

### Explanation

Separating a dataset helps determine whether a model can **generalize to data it hasn't seen during training**.

Typically:

- **Training set** — used to learn the model's parameters.
- **Validation set** — used during development to tune configurations and compare model versions.
- **Test set** — kept separate until the end and used to provide an unbiased estimate of final performance.

If the same data is used for both training and evaluation, the model may appear to perform very well simply because it has already seen that data. Separating the datasets helps identify this type of overfitting.

---

# Q9 — Model Selection

### Question

**What are the key factors affecting model selection?**  
*Choose all that apply.*

### Options

- [x] **Model accuracy.**
- [x] **Inference throughput.**
- [x] **Inference latency.**
- [x] **Computational cost associated with inference.**

### Correct Answers

**All of the above.**

### Explanation

Model selection isn't simply about choosing the model with the highest accuracy.

In a production environment, you also need to consider how efficiently the model can run.

- **Accuracy** determines how well the model performs the intended task.
- **Inference latency** measures how long the model takes to produce a result.
- **Inference throughput** measures how many inputs the system can process over a given period.
- **Computational cost** affects infrastructure requirements and the overall cost of operating the system.

For example, a slightly less accurate model may be more suitable for a real-time application if it has significantly lower latency and computational requirements.

---

# Q10 — Deep Learning for Flood Mapping

### Question

**How does deep-learning based image segmentation for flood mapping enable disaster management?**  
*Choose all that apply.*

### Options

- [x] **Quickly generate alerts for affected populations.**
- [x] **Perform accurate impact analysis and effective mitigation strategies.**
- [x] **Monitor and study the evolution of disaster events over time.**
- [x] **Assist with rapid response and recovery planning.**

### Correct Answers

**All of the above.**

### Explanation

Image segmentation can identify flooded areas at a **pixel level**, allowing systems to determine which parts of an image are affected by flooding.

When combined with satellite imagery and deep learning, this can provide valuable information much faster than relying entirely on manual analysis.

The resulting flood maps can support:

- Identifying affected regions.
- Assessing the scale and impact of flooding.
- Monitoring how the affected area changes over time.
- Supporting emergency response teams.
- Helping with recovery and mitigation planning.
- Providing information that can contribute to alerts and situational awareness.

This makes automated flood mapping particularly useful during large-scale disasters where conditions can change quickly.

---

# Quick Answer Key

| Question | Correct Answer |
|---|---|
| **Q1** | **4 — All of the above** |
| **Q2** | **2, 3, 4** |
| **Q3** | **1** |
| **Q4** | **1, 2, 3, 4** |
| **Q5** | **3** |
| **Q6** | **4** |
| **Q7** | **1, 2, 3** |
| **Q8** | **2** |
| **Q9** | **1, 2, 3, 4** |
| **Q10** | **1, 2, 3, 4** |

---

## Topics Covered

This assessment touches on several important areas of modern computer vision and AI:

- **Computer Vision fundamentals**
- **Image classification**
- **Object detection**
- **Image segmentation**
- **Satellite imagery and SAR**
- **Sentinel-1**
- **Machine learning workflows**
- **Transfer learning**
- **Data augmentation**
- **NVIDIA TAO Toolkit**
- **NVIDIA NGC**
- **Model evaluation and selection**
- **Inference performance**
- **Disaster management**
- **Deep-learning-based flood mapping**

The main theme across these questions is that building an effective AI solution involves more than just training a model. **Data quality, model selection, evaluation, optimization, deployment constraints, and the actual application of the model all matter.**