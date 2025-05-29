from DBConfig import DBConnection

import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def generate(data):
        
    pos_count = data.count('positive')
    neg_count = data.count('negative')
    neu_count = data.count('neutral')

   
    labels = ['positive', 'negative', 'neutral']
    sizes = [pos_count, neg_count, neu_count]

   
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Results')
    plt.axis('equal')  

   
    plt.legend()

    
    plt.tight_layout()
      
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

   
    graph = base64.b64encode(image_png).decode('utf-8')

    
    plt.close()

    return graph


    
if __name__ == '__main__':
    generate([])