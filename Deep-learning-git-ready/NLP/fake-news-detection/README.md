# Fake News Detection with Fine-Tuned RoBERTa

A binary text classifier built by fine-tuning [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) to label news articles as **Fake** or **Real** based on their text.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hasannlii0/Deep-learning/blob/main/NLP/fake-news-detection/fake_news_detection.ipynb)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Results

Fine-tuned for 1 epoch on a GPU runtime:

| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|:-----:|:--------------:|:----------------:|:--------------------:|
| 1 | 0.0015 | 0.000026 | 100.00% |

Example prediction:
```
Input: "Argentina has won 2022 world cup"
Predicted Class: Fake News
```

> **A note on the near-perfect accuracy:** this dataset is well known for containing an easy shortcut for models to exploit — its "real" articles come almost exclusively from Reuters and many retain wire-service framing (the notebook's own test example begins `"Washington (Reuters) - ..."`), while the fake articles don't share that style. A model can hit ~100% validation accuracy by learning this *source fingerprint* rather than genuine markers of misinformation. Treat this result as a strong fit to this specific dataset, not as a validated real-world fake-news detector — see [Notes & possible improvements](#notes--possible-improvements) below.

## Model architecture

A pretrained transformer encoder fine-tuned for sequence classification:

1. **Base model** — [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) (125M parameters), loaded via `AutoModelForSequenceClassification` with `num_labels=2`.
2. **Tokenizer** — RoBERTa's pretrained BPE tokenizer, with `padding="max_length"`, `truncation=True`, `max_length=256`.
3. **Fine-tuning** — full-model fine-tuning (no layers frozen) using the Hugging Face `Trainer` API.
4. **Classification head** — RoBERTa's built-in sequence classification head (pooled output → linear layer → 2 logits).

| Hyperparameter | Value |
|---|---|
| Base model | FacebookAI/roberta-base |
| Max sequence length | 256 |
| Learning rate | 2e-5 |
| Train / eval batch size | 16 |
| Epochs | 1 |
| Weight decay | 0.01 |
| Precision | fp16 |
| Optimizer | AdamW (Trainer default) |

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — 44,898 articles total: 23,481 fake + 21,417 real. Only the `text` column is used as input; `title`, `subject`, and `date` are dropped, and a binary `label` (0 = fake, 1 = real) is added.

The full dataset is shuffled (`random_state=42`) and split:

| Split | Rows | Share |
|---|---|---|
| Train | 31,428 | 70% |
| Validation | 6,735 | 15% |
| Test | 6,735 | 15% |

## Project structure

```
fake-news-detection/
├── fake_news_detection.ipynb   # Data prep, RoBERTa fine-tuning, evaluation, inference
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Hasannlii0/Deep-learning.git
cd Deep-learning/NLP/fake-news-detection
```

### 2. Create an environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for fine-tuning RoBERTa — the notebook uses `device_map="auto"` and `fp16=True`, both of which expect a GPU.

### 3. Get the dataset

The notebook downloads the dataset directly from Kaggle's API, which requires Kaggle credentials:

1. Create a Kaggle account and go to **Account → Create New API Token** to download `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows), and make sure it's not world-readable:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Either let the notebook's first cell download it via `curl`, or fetch it yourself:
   ```bash
   kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
   unzip fake-and-real-news-dataset.zip
   ```

This produces `Fake.csv` and `True.csv`, which the notebook expects in its working directory (the notebook was originally written for Google Colab, where files live under `/content/` — update those paths if you're running locally, e.g. to `./Fake.csv`).

### 4. Run the notebook

```bash
jupyter notebook fake_news_detection.ipynb
```

Or open it directly in Google Colab using the badge at the top of this README.

## Notes & possible improvements

- **Source-fingerprint leakage** — as noted above, the near-100% accuracy is likely inflated by a stylistic artifact (Reuters-style real articles vs. non-Reuters fake ones) rather than genuine misinformation detection. Stripping wire-service prefixes/bylines before training, or evaluating on an out-of-distribution set of real/fake articles from other sources, would give a more honest measure of generalization.
- **Single epoch** — the model converges to near-zero loss within 1 epoch on this dataset; more epochs would likely just overfit further. Early stopping or a smaller learning rate may be worth exploring on a harder dataset.
- **No learning-rate scheduling beyond `Trainer` defaults** — worth comparing against a linear or cosine schedule with warmup.
- **Colab-style paths** (`/content/...`) — swap these for relative paths if running locally.

## Acknowledgements

- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) on Kaggle, by Clément Bisaillon.
- Model: [RoBERTa](https://arxiv.org/abs/1907.11692) (Liu et al., 2019), via Hugging Face [`transformers`](https://github.com/huggingface/transformers).
- Built with Hugging Face `transformers`, `datasets`, and `evaluate`.

## License

This project is licensed under the MIT License — see the [repository LICENSE](../../LICENSE) for details.
