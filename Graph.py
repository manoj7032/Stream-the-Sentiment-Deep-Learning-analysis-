from DBConfig import DBConnection

import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def generate():
        
    db = DBConnection.getConnection()
    cursor = db.cursor()

 
    cursor.execute("SELECT * FROM evaluations")

    
    rows = cursor.fetchall()

    algorithms = []
    acc = []
    prec = []
    rec = []
    f1 = []

   
    for row in rows:
        algorithms.append(row[0])
        acc.append(float(row[1]))
        prec.append(float(row[2]))
        rec.append(float(row[3]))
        f1.append(float(row[4]))

    
    cursor.close()
    db.close()

    
    plt.figure(figsize=(10, 6))

    
    bar_width = 0.2
    positions = range(len(algorithms))

  
    plt.bar(positions, acc, width=bar_width, label="Accuracy", color='#cc00ff')
    plt.bar([p + bar_width for p in positions], prec, width=bar_width, label="Precision", color='#99ffff')
    plt.bar([p + 2 * bar_width for p in positions], rec, width=bar_width, label="Recall", color='#0099ff')
    plt.bar([p + 3 * bar_width for p in positions], f1, width=bar_width, label="F1-Score", color='#0033ff')

   
    plt.xlabel('Algorithms')
    plt.ylabel('Performance Metrics')
    plt.title('Algorithm Performance Comparison')

   
    plt.xticks([p + 1.5 * bar_width for p in positions], algorithms)

    
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
    generate()