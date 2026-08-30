# AI and data work

Coursework and personal projects in machine learning, data mining, and databases.
Each directory is self-contained with its own notebooks, data, and dependencies.

```
coursework/     graded work, one directory per course
projects/       standalone projects, not tied to a course
```

## coursework

| Directory | Course |
|---|---|
| `CS 324 Princ Software Engineering` | Software engineering principles |
| `CS329 Intro to DBMS` | Database systems, SQL |
| `CS428 Artificial Intelligence` | AI, search, neural networks |
| `CS522 Data Mining` | Classification, clustering, model evaluation |
| `CS 552 Deep Learning` | Deep learning, agent architectures |
| `CS200 Course Catalog Analysis` | Web scraping and descriptive analysis final project |

The strongest pieces here are `CS522 Data Mining/Final project/`, which runs logistic regression and random forest with SMOTE and GridSearchCV before comparing K-means against agglomerative clustering on silhouette and Davies-Bouldin scores, and `CS200 Course Catalog Analysis/`, which scrapes the Hood College catalog and compares curriculum structure across three departments.

## projects

### text-prediction

Next-word prediction with a Keras LSTM, trained on the True.txt split of the Kaggle Fake News dataset.
Includes a written explanation of how recurrent networks handle sequential data, a diagram of the RNN structure, and a terminal predictor that loads the saved tokenizer.

```bash
cd projects/text-prediction
pip install -r requirements.txt
python text_prediction.py
```

`prediction_model.ipynb` builds and trains the model; `text_prediction.py` loads the result and predicts from the terminal.
Neither the training corpus (`True.txt`) nor the saved model (`True_model.keras`) is committed, so the notebook must be run before the predictor has a model to load.
The notebook's paths point at Colab locations and need adjusting to run locally.

## What is mine and what is not

The assignment notebooks, final projects, and the text-prediction write-up are my own work.

`coursework/CS 552 Deep Learning/` contains the AIPython library by David Poole and Alan Mackworth, distributed under CC BY-NC-SA and included as course reference material.
My changes to it are limited to the agent controller files.
The text-prediction model follows a published tutorial, adapted to a different dataset and a three-word context window.

Lecture slides, syllabi, exam guidelines, and provided datasets throughout `coursework/` were supplied by the courses and belong to their authors.
`projects/text-prediction/LICENSE` covers that project only; the repository as a whole is not under a single licence.

## Related repositories

Two pieces of CS428 work were developed past the assignment and live on their own:

- [ASL-Recognition-with-CNN](https://github.com/danielcoblentz/ASL-Recognition-with-CNN) - gesture recognition with a custom CNN
- [Image-classification](https://github.com/danielcoblentz/Image-classification) - VGG16 transfer learning

Stale partial copies of both remain under `coursework/CS428 Artificial Intelligence/`; the repositories above are the current versions.

## Running the notebooks

Notebooks are written for Google Colab and Jupyter and install their own dependencies, so there is no shared environment file.
Several read data from paths relative to their own directory, so open them from where they sit rather than from the repository root.
