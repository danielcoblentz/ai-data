# Coursework archive - AI, data mining, and deep learning

Notebooks, assignments, and course materials from five Hood College computer science courses.
This is a personal archive rather than a single project.

| Directory | Course |
|---|---|
| `CS 324 Princ Software Engineering` | Software engineering principles |
| `CS329 Intro to DBMS` | Database systems, SQL |
| `CS428 Artificial Intelligence` | AI, search, neural networks |
| `CS522 Data Mining` | Classification, clustering, model evaluation |
| `CS 552 Deep Learning` | Deep learning, agent architectures |
| `CS200 Course Catalog Analysis` | Web scraping and descriptive analysis final project |

## What is mine and what is not

My own work is the assignment notebooks and final projects, most notably:

- `CS522 Data Mining/Final project/` - logistic regression and random forest with SMOTE and GridSearchCV, then K-means and agglomerative clustering evaluated with silhouette and Davies-Bouldin scores
- `CS200 Course Catalog Analysis/` - scrapes the Hood College catalog and compares curriculum structure across three departments
- `CS329 Intro to DBMS/` - SQL notes and exercises

The `CS 552 Deep Learning` directory contains the AIPython library by David Poole and Alan Mackworth, distributed under CC BY-NC-SA and included here as course reference material.
My changes to it are limited to the agent controller files.

Lecture slides, syllabi, exam guidelines, and provided datasets throughout this repository were supplied by the courses and belong to their respective authors.

## Related repositories

Two pieces of CS428 work were developed past the assignment and live on their own:

- [ASL-Recognition-with-CNN](https://github.com/danielcoblentz/ASL-Recognition-with-CNN) - gesture recognition with a custom CNN
- [Image-classification](https://github.com/danielcoblentz/Image-classification) - VGG16 transfer learning

Stale partial copies of both remain under `CS428 Artificial Intelligence/`; the repositories above are the current versions.

## Running the notebooks

Notebooks are written for Google Colab and Jupyter and install their own dependencies, so there is no shared environment file.
Several read data from paths relative to their own directory, so open them from where they sit rather than from the repository root.
