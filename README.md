# News Classification Project

This project focuses on classifying news articles into various categories using a deep learning model. It encompasses the entire pipeline from data collection and preprocessing to model training and evaluation.

## Project Structure and Workflow

### 1. Data Crawling

The `Crawler/` directory contains scripts responsible for collecting news article URLs and their content.

- `crawler.py`: This script provides the core functionality for crawling web data.
- Other Python files within `Crawler/` (e.g., `collecting_bus_fin_news.py`, `collecting_health_medical_new.py`, `collecting_sport_new.py`, `collecting_technology_news.py`): These specialized scripts are used to crawl data for four distinct categories: Business/Finance, Health/Medical, Sport, and Technology.

All crawled URLs are initially saved into respective text files within the `URL/` directory.

### 2. Metadata Extraction and Preprocessing

After the URLs are collected, the `Metadata/` directory handles the extraction of relevant information.

- The saved URLs from `URL/` are processed to extract metadata, which includes assigning labels to the articles and extracting their content.
- The extracted metadata is then saved into a single JSON file named `metadata.json`, located in the `Metadata/` directory.

### 3. Data Splitting

For robust model training and evaluation, the prepared data is split into training, validation, and test datasets.

- The `train_val_test/` directory stores these split datasets:
    - `train_data.json`: Contains the data used for training the model.
    - `val_data.json`: Contains the data used for validating the model during training.
    - `test_data.json`: Contains the data used for final model evaluation.

### 4. Model Training and Evaluation

The `model/` directory houses the scripts for building, training, and testing the news classification model.

- `roberta.py`: This script contains the implementation for training and testing the model.
- The project utilizes a fine-tuned pre-trained RoBERTa model for classification.
- The model has achieved an accuracy of 100% on the test dataset, demonstrating its effectiveness in classifying news articles.

## Execution Environment

This project was trained and tested on Google Colab to leverage GPU resources for faster model training. You can access the Google Colab notebook here: [News Classification Colab Notebook](https://colab.research.google.com/drive/1NYTvM8Vm3ZZ8eKjbubpmZYeWet4PJeN9?usp=sharing)

## Installation

To set up the project locally, you will need to install the required Python packages. These can be found in the `requirements.txt` file.

```bash
pip install -r requirements.txt
``` 