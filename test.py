import torch
from transformers import BertTokenizer, BertForSequenceClassification
from src.model import PatentSentenceClassifier
import matplotlib.pyplot as plt
import pandas as pd
import time

from src.config import load_config

def main(test_path):

    start_time = time.time()

    # Load config file
    cfg = load_config('config.yaml')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load Tokenizer and Base Model
    bert_tokenizer = BertTokenizer.from_pretrained(cfg.model.name)
    base_model = BertForSequenceClassification.from_pretrained(cfg.model.name, num_labels=cfg.model.num_lables)
    print(f"\nTokenizer and Base Model loaded succesfully. Using: '{cfg.model.name}'")

    # Load Finetuned Patent Model
    loaded_model = PatentSentenceClassifier.load_from_checkpoint(cfg.model.checkpoint, model=base_model, tokenizer=bert_tokenizer)
    loaded_model.eval()
    loaded_model.to(device)
    print(f"\nModel loaded succesfully. Using: '{cfg.model.checkpoint}'")

    # Perform inference iterating over each text input
    df = pd.read_excel(test_path)
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
    
    end_time = time.time()
    testing_time = end_time - start_time
    print(f"Testing time: {testing_time:.2f} sec")

    # Convert results to a DataFrame 
    results_df = pd.DataFrame(results)
    merged_df = pd.merge(df, results_df, on='sent_id', how='right') # merge with original dataframe
    output_path = f'{cfg.test.save_dir}/{cfg.test.save_name}.xlsx'
    merged_df.to_excel(output_path, index=False)
    print(f"\nResults saved successfully to: '{output_path}'")

if __name__ == "__main__":
    cfg = load_config('config.yaml')
    test_path = cfg.data.test_path
    #test_path = '/app/patents/CN214409472U.xlsx' # set input path to test set
    #test_path = '/app/data/test_agreement.xlsx' # set input path to test set
    main(test_path)

