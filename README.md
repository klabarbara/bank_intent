
Using this file during dev to document my thoughts and findings

Aug 8 2025

The semantic meaning of labels in this dataset (eg: "request_refund") is essential to feed into the classifier, but integer labels are useful for effciciency and convention when the semantics are not relevant (eg: when calculating accuracy metrics). Mapping class integer labels to their english counterparts. 

### Sample Results from Dev Runs

**Zero-shot classification on test set**

| Model                                      | Accuracy | Macro-F1 | Runtime |
| ------------------------------------------ | -------- | -------- | ------- |
| `typeform/distilbert-base-uncased-mnli`    | 0.138    | 0.152    | 6 min   |
| `facebook/bart-large-mnli`                 | 0.311    | 0.316    | 30 min  |
