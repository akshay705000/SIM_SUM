
# SIMSUM: Document-level Text Simplification via Simultaneous Summarization

## Project Overview

This repository contains the implementation of **SIMSUM**, a model designed for document-level text simplification through simultaneous summarization. The project builds on Transformer-based architectures, specifically adapting BART and T5 models to simplify long documents while preserving semantic accuracy. We introduce an enhanced loss function incorporating a simplicity metric, ensuring a balance between simplicity and meaning retention. Additionally, we use **TextRank** for keyword extraction, which serves as a prompt to boost summarization and simplification performance.

## Key Contributions
- **Transformer-based Model Adaptation**: We adapted the SIMSUM model using BART and T5 as the backbone models for document-level simplification.
- **Enhanced Loss Function**: Introduced a simplicity metric within the loss function to balance between simplicity and semantic accuracy during training.
- **TextRank for Keyword Extraction**: Employed the TextRank algorithm to extract keywords and use them as prompts, improving summarization accuracy and coherence.
  
## Repository Structure
```
├── SimSum/ 
     ├── data/    
     |     ├── D-Wiki/                # D-Wikipedia dataset
     |     ├── wiki_doc/              # WikiDoc dataset         
     ├── Bart2.py                     # SIMSUM model with BART backbone
     ├── T5_2.py                      # SIMSUM model with T5 backbone
     ├── Bart_baseline_finetuned.py   # Baseline BART model for comparison
     |── T5_baseline_finetuned.py     # Baseline T5 model for comparison
     ├── evaluate.py                  # Script for automatic evaluation
     ├── main.py                      # Script for training the model
     ├── requirements.txt             # Python dependencies
     └── SIM                         # Figures and visual resources


## Datasets

The datasets used for this project are located in the `SimSum/data/` directory:
- **D-Wikipedia (D-Wiki)**: A simplified version of Wikipedia articles.
- **WikiDoc**: A dataset with long-form documents from Wikipedia.

## Installation

Execute the NLP_SIMSUM_Project.ipynb file
```

## Training the Model

To train the SIMSUM model with either BART or T5 as the backbone, execute the following command:

```bash
python main.py
```

For single-model training, you can use the following scripts:
- BART: `Bart_baseline_finetuned.py`
- T5: `T5_baseline_finetuned.py`

## Evaluation

### Automatic Evaluation
To evaluate the model using standard metrics (SARI, D-SARI, BLEU, FKGL), run:

```bash
python evaluate.py
```

## Evaluation

To completely train and evaluate

```bash
NLP_SIMSUM_Project.ipynb
```

### Human Evaluation
For human evaluation, manually assess the generated simplified documents for readability, simplicity, and content retention.

## Results
- **SARI**: Measures how well the model simplifies the text.
- **D-SARI**: A variant of SARI for document-level simplification.

## Future Enhancements

- Fine-tuning the model with more diverse datasets for better generalization.
- Experimenting with different keyword extraction techniques to improve summarization accuracy.
- Extending the simplicity metric to adapt to different document genres.
