 
# Emotion Detection from Text

This project uses deep learning (LSTM + Bidirectional LSTM) to detect human emotions from text, such as sadness, joy, anger, love, fear, and surprise.

![Model Accuracy and Loss](images/training_plots.png)

## Demo

A live demo is available via Streamlit app:

```bash
streamlit run app/streamlit_app.py

Model Architecture

Tokenization & Padding
Embedding layer
Bidirectional LSTM + LSTM
Dense layers with softmax output
Achieved:

Train Accuracy: 98%
Validation Accuracy: 93%
Loss: < 0.05 (train), < 0.2 (val)
Dataset

We used the emotion dataset from Hugging Face Datasets: https://huggingface.co/datasets/dair-ai/emotion

Classes:
sadness
joy
love
anger
fear
surprise

Evaluation

Confusion Matrix (Test Set)