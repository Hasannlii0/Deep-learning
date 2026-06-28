# Deep Learning

A growing collection of deep learning projects, organized by domain. Each project is self-contained — its own notebook, README, and dependencies — so you can clone the whole repo or just `cd` into the one project you care about.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/built%20with-PyTorch-EE4C2C.svg)

## Projects

| Domain | Project | Description | Result |
|---|---|---|---|
| NLP | [AG News Classification](NLP/ag-news-classification) | 4-class news topic classification with a Transformer encoder and BPE tokenizer **built from scratch** (no pretrained models) | 90.07% test accuracy |
| NLP | [Fake News Detection](NLP/fake-news-detection) | Binary fake/real news classification by fine-tuning a pretrained **RoBERTa-base** | 100.00% validation accuracy* |

\* See the [results caveat](NLP/fake-news-detection/README.md#results) — this reflects a known leakage issue in that dataset, not a claim of solved fake-news detection.

More domains (Computer Vision, Generative Models, etc.) may be added over time, each as its own top-level folder following the same project layout.

## Repository structure

```
Deep-learning/
├── NLP/
│   ├── README.md                          # Index of NLP projects
│   ├── ag-news-classification/
│   │   ├── AG_news_classification.ipynb
│   │   ├── README.md
│   │   └── requirements.txt
│   └── fake-news-detection/
│       ├── fake_news_detection.ipynb
│       ├── README.md
│       └── requirements.txt
├── .gitignore
├── LICENSE
└── README.md                              # You are here
```

Each project folder is independent: its own notebook, its own `requirements.txt` (dependencies differ between projects — e.g. one trains a tokenizer from scratch, the other fine-tunes a Hugging Face model), and its own `README.md` with full setup instructions, architecture notes, and results.

## Getting started

Clone the full repo:

```bash
git clone https://github.com/Hasannlii0/Deep-learning.git
cd Deep-learning
```

Then move into whichever project interests you and follow its own README — for example:

```bash
cd NLP/fake-news-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook fake_news_detection.ipynb
```

Most notebooks were originally written for Google Colab and download their datasets directly from the Kaggle API — see each project's README for Kaggle credential setup.

## Contributing

This is primarily a personal learning and portfolio repo, but issues, suggestions, and pull requests (bug fixes, clearer explanations, additional experiments) are welcome.

## License

This repository is licensed under the MIT License — see [LICENSE](LICENSE) for details.
