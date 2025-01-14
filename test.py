import torch
from transformers import BertTokenizer, BertForSequenceClassification
from src.model import PatentClassifier
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config

def main(input_path, output_filename = 'results_new'):

    # Load config file
    cfg = load_config('config.yaml')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load Tokenizer and Base Model
    bert_tokenizer = BertTokenizer.from_pretrained(cfg.model.name)
    base_model = BertForSequenceClassification.from_pretrained(cfg.model.name, num_labels=cfg.model.num_lables)
    print('\nTokenizer and Base Model loaded succesfully.')

    # Load Finetuned Patent Model
    checkpoint_path = '/app/models/best-checkpoint.ckpt'
    loaded_model = PatentClassifier.load_from_checkpoint(checkpoint_path, model=base_model, tokenizer=bert_tokenizer)
    loaded_model.eval()
    loaded_model.to(device)
    print('\nModel loaded succesfully.')

    # Perferom inference iterating over each text input
    df = pd.read_excel(input_path)
    results = []
    for idx, text in enumerate(df['sent']):
        try:
            inputs = bert_tokenizer(text, truncation=True, padding=True, max_length=cfg.model.max_length, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            int_to_label = {0: 'FUN', 1: 'STR', 2: 'MIX', 3: 'OTH'}
            with torch.no_grad():
                outputs = loaded_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
            
            result = {
                'sent_id': df['sent_id'].iloc[idx],
                'predicted_class': predicted_class.item(),
                'predicted_tag': int_to_label[predicted_class.item()],
                'probabilities': [round(prob, 2) for prob in probabilities[0].tolist()]
            }
            
            results.append(result)
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.append({'text': text, 'error': str(e)})

    # Convert results to a DataFrame 
    results_df = pd.DataFrame(results)
    merged_df = pd.merge(df, results_df, on='sent_id', how='right') # merge with original dataframe
    merged_df.to_excel(f'/app/results/{output_filename}.xlsx', index=False)

if __name__ == "__main__":
    input_path = '/app/data/CN214409472U.xlsx' # set input path
    output_filename = 'results_new' # set name of the output file
    main(input_path, output_filename)

