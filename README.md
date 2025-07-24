# MISS-QA: A Multimodal Scientific Information-Seeking QA Benchmark

<p align="center">
  <a href="https://github.com/yilunzhao/MISS-QA">🌐 Github</a> •
  <a href="https://arxiv.org/abs/2507.10787">📖 Paper</a> •
  <a href="https://huggingface.co/datasets/yale-nlp/MISS-QA">🤗 Data</a>
</p>

MISS-QA (Multimodal Information-Seeking over Scientific papers – Question Answering) is the **first benchmark** specifically designed to evaluate the ability of multimodal foundation models to **interpret schematic diagrams** and answer **information-seeking questions** within scientific literature.

> 🔬 “Can Multimodal Foundation Models Understand Schematic Diagrams?”

------

## 📰 News

[May 15, 2025] MISSQA has been accepted to Findings of ACL 2025!

## 🌟 Highlights

- 📚 **1500 QA pairs** annotated by **expert researchers**
- 📄 Covers **465 AI-related papers** from arXiv
- 🎯 Focuses on **schematic diagrams**, not just charts or tables
- 🤖 Evaluates **18 frontier vision-language models** (o4-mini, Gemini-2.5-Flash, and Qwen2.5-VL)
- 🧠 Automatic evaluation protocol trained on **human-scored data**

------

## 🧩 Benchmark Structure

Each example in MISS-QA includes:

- A **schematic diagram** from a scientific paper
- A **highlighted visual element** (bounding box)
- A **free-form information-seeking question**
- The corresponding **scientific context**
- A human-annotated **answer** (or marked as unanswerable)

### 🔍 Information-Seeking Scenarios

- **Design Rationale**
- **Implementation Details**
- **Literature Background**
- **Experimental Results**
- **Other** (e.g., limitations, ethics)

------

## 📊 Model Evaluation

MISS-QA is used to benchmark proprietary and open-source **multimodal foundation models**. Performance is automatically scored using a custom evaluation protocol aligned with human judgment.

------

## 🛠️ How to Use

### 🔁 Step 0: Installation

```bash
git clone https://github.com/yilunzhao/MISSQA.git
cd MISSQA
conda create --name missqa python=3.10
conda activate missqa
pip install -r requirements.txt
```

### 🔁 Step 1: Download Dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/yale-nlp/MISS-QA
```

### 🔁 Step 2: Run Model Inference

Use the provided bash script to run inference with your multimodal model:

```bash
bash scripts/vllm_large.sh
```

This will generate model responses and save them to:

```swift
./outputs/
```

### ✅ Step 3: Evaluate Model Accuracy

Once inference is complete, run the accuracy evaluation script:

```bash
python acc_evaluation.py
```

The processed and scored outputs will be saved to:

```swift
./processed_outputs/
```

## ✍️ Citation

If you use our work and are inspired by our work, please consider cite us:

```
@inproceedings{zhao-etal-2025-multimodal-foundation,
  title     = {Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers},
  author    = {Zhao, Yilun and Wang, Chengye and Li, Chuhan and Cohan, Arman},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  year      = {2025},
  month     = jul,
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  pages     = {18598--18631},
  url       = {https://aclanthology.org/2025.findings-acl.957/}
}
```

