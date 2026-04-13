# 📊 Math Data Cleaning Pipeline for LLM Post-Training

This project implements a **data cleaning and structuring pipeline** for math reasoning datasets, designed for post-training small language models such as **Qwen3-0.6B**.

The pipeline transforms raw unstructured data into high-quality supervised training samples.

---

## 🚀 Features

- 🔍 Rule-based information extraction
- 🤖 Model-assisted extraction using local Qwen model
- 🧹 Noise removal and text normalization
- ✅ Quality scoring and filtering
- 🔁 Deduplication (exact + normalized)
- 📊 Data categorization (math type & difficulty)
- 🧪 Automatic sampling for manual inspection

---

## 🧠 Pipeline Overview

```text
Raw Data
   ↓
Text Cleaning
   ↓
Rule-based Extraction
   ↓
Qwen-assisted Completion
   ↓
Quality Filtering
   ↓
Deduplication
   ↓
Structured Training Data
```
---

## 📂 Output Format

Each processed sample is structured as:

```json
{
  "problem": "...",
  "solution": "...",
  "final_answer": "...",
  "quality_score": 8,
  "math_bucket": "algebra",
  "difficulty": "medium"
}
