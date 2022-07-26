# Roberta for SNLI dataset

## Dataset Summary
The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE).

## Data Instances
For each instance, there is a string for the premise, a string for the hypothesis, and an integer for the label. Note that each premise may appear three times with a different hypothesis and label. See the SNLI corpus viewer to explore more examples.

```
{'premise': 'Two women are embracing while holding to go packages.'
 'hypothesis': 'The sisters are hugging goodbye while holding to go packages after just eating lunch.'
 'label': 1}
```

## Download Datasets

```
$ bash data.sh
```


## Preprocessing dataset

```
$python preprocess.py

```

## Train and evaluate model

```
$ python snli.py 
```

## Accuracy
```
91.64

```
