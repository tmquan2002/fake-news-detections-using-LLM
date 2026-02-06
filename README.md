# Fake News Detection: PLMs vs. LLMs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

This repository contains the source code and experimental setup for the Term Paper: **"A Survey on the Use of Large Language Models in Fake News Detection"**.

## Project Overview
The project compares two distinct AI paradigms for detecting disinformation:
1.  **Discriminative Approach (RoBERTa):** A fine-tuned PLM (Pre-trained Language Model) acting as a high-speed classifier.
2.  **Generative Approach (Gemini 1.5 Flash):** A zero-shot LLM (Large Language Model) acting as an explainable fact-checker using Chain-of-Thought (CoT) reasoning.

---

## Getting Started (How to Clone & Run)

Follow these steps to set up the project locally on your machine.

### 1. Prerequisites
* **Python 3.8** or higher.
* **Git** installed.
* A **Google Gemini API Key** (Get it for free [here](https://aistudio.google.com/app/apikey)).

### 2. Clone the Repository
Open your terminal/command prompt and run:

```bash
# Clone this repository
git clone https://github.com/tmquan2002/fake-news-detections-using-LLM

# Navigate into the project folder
cd fake-news-detections-using-LLM