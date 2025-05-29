from DBConfig import DBConnection

class StoreData:


    def testingresults(algo, d):
        db = DBConnection.getConnection()
        cursor = db.cursor()
        cursor.execute("delete from evaluations where algorithm='"+algo+"'")
        
        sql = "insert into evaluations values(%s,%s,%s,%s,%s)"

        values = (algo, str(d[0]), str(d[1]), str(d[2]), str(d[3]))
        cursor.execute(sql, values)


        db.commit()
        
          
    
if __name__ == '__main__':
    StoreData.testingresults('LSTM', (1,1,1,1))

    


