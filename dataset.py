import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from io import StringIO



import io
import base64
import numpy as np

def main():
   
    df = pd.read_csv("comments.csv",encoding='latin-1')
    buffer = StringIO()
    
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)

    df_head = df.head()
    df_html = df_head.to_html(classes='table table-striped', index=False)


   
    sentiment_counts = df['Sentiment'].value_counts()

    
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', labels=['Positive(2)', 'Neutral(1)', 'Negative(0)'], colors=['green', 'orange', 'red'])
    plt.title('Sentiment Distribution')
    plt.ylabel('')
    
       
    plt.legend()

  
    plt.tight_layout()
        
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graph = base64.b64encode(image_png).decode('utf-8')

   
    plt.close()

    return info_str, df_html, graph


if __name__ == '__main__':
    main()
