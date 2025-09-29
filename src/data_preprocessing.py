import pandas as pd
import os 
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir,exist_ok = True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """Transform the text by removing punctuation, stopwords and applying stemming"""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    text = [word for word in text if word.isalnum()]
    
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_df(df,text_column = 'text',target_column = 'target'):
    """Preprocess the dataframe by transforming text and encoding target labels"""
    try:
        logger.debug('Starting text transformation')
        
        lr = LabelEncoder()
        df[target_column] = lr.fit_transform(df[target_column])
        logger.debug('Target column encoded')
        
        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicates removed from dataframe')
        
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text transformation completed')
        return df
    except KeyError as e:
        logger.error('Column not found during preprocessing: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured during normalization: %s',e)
        raise

def main(text_column = 'text',target_column = 'target'):
    
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Train and Test data loaded')
        
        train_processed = preprocess_df(train_data,text_column,target_column)
        test_processed = preprocess_df(test_data,text_column,target_column)
        
        data_path  = os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)
        
        train_processed.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)
        logger.debug('Processed Train and Test data saved to %s',data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s',e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data file: %s',e)
        raise
    except Exception as e:
        logger.error('failed to complete data transformation: %s',e)
        print(f'Error: {e}')

if __name__== '__main__':
    main()
        