
Using this file during dev to document my thoughts and findings

Aug 8 2025

The semantic meaning of labels in this dataset (eg: "request_refund") is essential to feed into the classifier, but integer labels are useful for effciciency and convention when the semantics are not relevant (eg: when calculating accuracy metrics). Mapping class integer labels to their english counterparts. 


Zero-shot classification on test set:

[30:02<00:00, 28.62s/it]
Baseline acc=0.311 macro-F1=0.316

