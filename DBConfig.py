

import mysql.connector
class DBConnection:
    @staticmethod
    def getConnection():
        database = mysql.connector.connect(host="localhost", user="root", passwd="70324", port='3306', db='youtubecomments')
        return database
if __name__=="__main__":
    print(DBConnection.getConnection())