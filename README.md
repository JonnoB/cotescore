# Get your COTe

ocument layout analysis (DLA) is a key elements of Document Understanding. Traditional metrics used in this process, such as IoU, F1, and  mAP do not provide the insight needed for systematic model improvement or selection. In addition, the assumptions underlying these metrics make it challenging to compare models and datasets. To encourage more robust, comparable, and nuanced Document Understanding, we introduce: The Structural Semantic Unit (SSU) a relational labelling approach to making DLA measurement robust to changes in ground truth granularity and the Coverage, Overlap, Trespass, and Excess (COTe) score, a decomposable metric for measuring page parsing quality. This decomposability enables detailed model evaluation and debugging, substantially reducing issues when evaluating models trained with different labelling granularity. We demonstrate the value of these methods through case studies, and by evaluating 5 common DLA models on 3 DLA datasets, analysing the differences between our metrics and traditional approaches. We show that, across a variety of media, mAP can yield misleading or uninterpretable results due to its failure to penalise critical layout errors.
Finally, we release an SSU labelled dataset and a Python library for applying COTe in Document Understanding projects. 





# Character Distribution Distance

The Character Error Rate (CER) and Word Error Rate (WER), are critical mainstays of evaluating the quality of Optical Character Recognition (OCR). However, these metrics assume that text has been perfectly parsed, which is often not the case. Under page parsing errors, CER and WER become undefined, limiting them to a role of local metric, making evaluating page-level OCR challenging. We introduce the Character Distribution Distance (CDD) and Word Distribution Distance (WDD), two page-level metrics that integrate both parsing and OCR errors into a single value. In addition, these metrics decompose into parsing and OCR error components. This decomposability allows practitioners to focus on the part of the Document Understanding (DU) pipeline that will have the most impact on the overall quality of text extraction. We demonstrate this approach using both case studies and data analysis. We test the CDD using Euclidean, Cosine, and Jensen-Shannon distances. We show that the metrics can be used to evaluate DU quality when text layout information is not available. We also show that although, for optimal triage, the CDD/WDD requires character-level positioning, the triaging only loses a small fraction of accuracy when bounding boxes are drawn at the line level. Remarkably, we show that the CDD can also show the parsing quality of end-to-end OCR models, which do not have explicit page parsing. We provide the CDD and WDD code as part of a python library to support DU research.

# extra dependencies for analysis

marimo
jinja2
plotnine