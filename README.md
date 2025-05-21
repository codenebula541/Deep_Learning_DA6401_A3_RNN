# Deep_Learning_DA6401_A3_RNN

##  Hindi Transliteration using RNNs with Attention

This project implements a neural transliteration system that converts Hindi words written in Latin script into the Devanagari script using Recurrent Neural Networks (RNNs). It explores the strengths and limitations of vanilla Seq2Seq models and how attention mechanisms improve performance on complex sequences.  
wandb project report link: https://wandb.ai/saurabh541-indian-institute-of-technology-madras/transliteration_model-2/reports/Assignment-3--VmlldzoxMjg4MDE4Nw

---

##  Goals of the Project

-  Model sequence-to-sequence transliteration using Recurrent Neural Networks.
-  Compare RNN cells: vanilla RNN, LSTM, GRU.
-  Investigate attention mechanisms in overcoming limitations of basic Seq2Seq models.
-  Visualize how attention guides the decoding process, token by token.

---

##  Models Implemented

-  Vanilla Seq2Seq with GRU
-  Vanilla Seq2Seq with LSTM
-  Attention-based Seq2Seq (Luong Attention)

---

##  Dataset

- **Source:** [Dakshina Dataset by Google](https://github.com/google-research-datasets/dakshina)
- **Task:** Transliterate Hindi words written in Latin script (e.g., `ajanabee`) to Devanagari script (e.g., `अजनबी`)
- **Format:** Tab-separated files with source and target sequences
- **Splits:**
  - `train.tsv`
  - `dev.tsv`
  - `test.tsv`

---

##  Key Features

- Character-level RNN encoder-decoder
- Luong-style attention implementation
- Visual attention heatmaps and animated GIFs
- Font support for Devanagari in visualizations
- Automated hyperparameter search with Weights & Biases (W&B)
- Word-level accuracy computation

---

##  Results

| Model              | Test Accuracy |
|-------------------|---------------|
| Vanilla Seq2Seq   | 39.30%        |
| Attention-based   | **40.50%**    |

 **Insight:** Attention model improved accuracy by learning to focus on relevant input characters, especially for longer or ambiguous words.

---

### 🔍 Example: Error Fixed by Attention

| Latin Input | True Output | Vanilla Output | Attention Output | Improved |
|-------------|-------------|----------------|------------------|----------|
| **ankor**   | अंकोर       | अंकोर          | **अंकोर**        | ✅        |
| **ankit**   | अंकित       | अंकित          | **अंकित**        | ✅        |
| **angarji** | अँग्रेज़ी   | अय्राजी        | **अँग्रजी**       | ✅        |

---

## 🚀 How to Run

-  Run the entire pipeline on [Kaggle Notebooks](https://www.kaggle.com)
-  Log all runs and metrics to Weights & Biases
-  Train using W&B Sweep with Bayesian Optimization
-  Evaluate final model on the test set
-  Generate 3×3 attention heatmaps and animated attention GIFs

---


## 📜 License

This project uses the open-source [Dakshina dataset](https://github.com/google-research-datasets/dakshina) released under the CC BY 4.0 License.

---
