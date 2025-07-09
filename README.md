# MISS-QA: A Multimodal Scientific Information-Seeking QA Benchmark

MISS-QA (Multimodal Information-Seeking over Scientific papers – Question Answering) is the **first benchmark** specifically designed to evaluate the ability of multimodal foundation models to **interpret schematic diagrams** and answer **information-seeking questions** within scientific literature.

> 🔬 “Can Multimodal Foundation Models Understand Schematic Diagrams?”

------

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
git clone https://github.com/QDRhhhh/MISSQA.git
cd MISSQA
conda create --name missqa python=3.10
conda activate missqa
pip install -r requirements.txt
```

### 🔁 Step 1: Run Model Inference

Use the provided bash script to run inference with your multimodal model:

```bash
bash scripts/vllm_large.sh
```

This will generate model responses and save them to:

```swift
./outputs/
```

### ✅ Step 2: Evaluate Model Accuracy

Once inference is complete, run the accuracy evaluation script:

```bash
python acc_evaluation.py
```

The processed and scored outputs will be saved to:

```swift
./processed_outputs/
```
