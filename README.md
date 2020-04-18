# Princeton Prosody Archive: Item Classification and Related Work

## Classification
Code for item classification is found in `classification/prosody_classifier.py`. 

Results for different classifier types:
Single-layer neural network (NN): 63% accuracy
Naive Bayes classifier: 87% accuracy
Support Vector Machine (SVM): 89% accuracy

Run `prosody_classifier.py` for SVM results. The trained NN is in `models/prosody_classifier.pt`.

## Topic Modelling
Code for topic modeling is found under `topic_modeling`. Word clouds are zipped under `topic_modeling/results`.

## ID Extraction
Code for extracting IDs from text is found in `id_extraction/id_extraction.py`.

## Authors and Acknowledgment
This project is the work of Eve Fleisig and was created for the Princeton Prosody Archive in 2019-2020.