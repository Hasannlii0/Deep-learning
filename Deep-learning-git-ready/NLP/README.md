# NLP

Natural language processing projects, ranging from a from-scratch Transformer to fine-tuning a pretrained model.

| Project | Task | Approach | Result |
|---|---|---|---|
| [`ag-news-classification/`](ag-news-classification) | 4-class news topic classification | Transformer encoder + BPE tokenizer, **built from scratch** | 90.07% test accuracy |
| [`fake-news-detection/`](fake-news-detection) | Binary fake/real news classification | Fine-tuned pretrained **RoBERTa-base** | 100.00% validation accuracy* |

\* see the [caveat on dataset leakage](fake-news-detection/README.md#results) in that project's README — the score reflects how easy this particular dataset is, not a generalization guarantee.

Each subfolder is self-contained: its own notebook, `README.md`, and `requirements.txt`. See the project README for setup details, dataset links, and architecture notes.
