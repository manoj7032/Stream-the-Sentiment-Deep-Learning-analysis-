from flask import Flask, render_template, request,flash
from flask import Response
from flask import session
import numpy as np
import sys
import os
import io
import random
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from DBConfig import DBConnection

app = Flask(__name__)
app.secret_key = "abc"

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/user")
def user():
    return render_template("user.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/newuser")
def newuser():
    return render_template("register.html")



@app.route("/user_register",methods =["GET", "POST"])
def user_register():
    try:
        sts=""
        name = request.form.get('name')
        pwd = request.form.get('pwd')
        mno = request.form.get('mno')
        email = request.form.get('email')
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from user_signup where email='" + email + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            sts = 0
        else:
            sql = "insert into user_signup values(%s,%s,%s,%s)"
            values = (name,email, mno,pwd)
            cursor.execute(sql, values)
            database.commit()
            sts = 1

        if sts==1:
            return render_template("user.html", msg="Registered Successfully..! Login Here.")


        else:
            return render_template("register.html", msg="User name already exists..!")



    except Exception as e:
        print(e)

    return ""



@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from user_signup where email='" + uid + "' and password='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid

            return render_template("user_home.html",uid=uid)
        else:

            return render_template("user.html", msg2="Invalid Credentials")

        return ""

@app.route("/userhome")
def userhome():
    return render_template("user_home.html")


@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin@gmail.com" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin.html",msg="Invalid Credentials")

@app.route("/adminhime")
def adminhome():
    return render_template("admin_home.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/datasetanalysis", methods =["GET", "POST"])
def datasetanalysis():
    
    from dataset import main
    txt, head, img=main()
    return render_template("datasetresults.html", txt=txt, head=head, graph=img)

@app.route("/classification")
def classification():
    return render_template("classification.html")


@app.route("/lstm", methods =["GET", "POST"])
def lstm():
    
    from Train_LSTM import lstm_model_training
    d=lstm_model_training()

    from StoreData import StoreData
    StoreData.testingresults('LSTM', d)
    
    return render_template("classification.html", msg="done")



@app.route("/bilstm", methods =["GET", "POST"])
def bilstm():
    
    from Train_Bi_LSTM import bilstm_model_training
    d=bilstm_model_training()

    from StoreData import StoreData
    StoreData.testingresults('Bi-LSTM', d)
    
    return render_template("classification.html", msg="done")




@app.route("/gru", methods =["GET", "POST"])
def gru():
    
    from Train_GRU import gru_model_training
    d=gru_model_training()

    from StoreData import StoreData
    StoreData.testingresults('GRU', d)
    
    return render_template("classification.html", msg="done")



@app.route("/rnn", methods =["GET", "POST"])
def rnn():
    
    from Train_RNN import rnn_model_training
    d=rnn_model_training()

    from StoreData import StoreData
    StoreData.testingresults('RNN', d)
    
    return render_template("classification.html", msg="done")


@app.route("/results")
def results():
    from Graph import generate
    graph=generate()

    database = DBConnection.getConnection()
    cursor2 = database.cursor()
    
    sql= "select * from evaluations "
    cursor2.execute(sql)
    records = cursor2.fetchall()
    return render_template("results.html", data=records, graph=graph)


@app.route("/search")
def search():
    return render_template("search.html")


@app.route("/history")
def history():
    database = DBConnection.getConnection()
    cursor2 = database.cursor()
    uid=session['uid']
    
    sql= "select * from videos where email=%s "
    values=(uid,)
    cursor2.execute(sql, values)
    records = cursor2.fetchall()
    return render_template("history.html", data=records)

@app.route('/historydetails/<int:v_id>')
def historydetails(v_id):
    database = DBConnection.getConnection()
    cursor2 = database.cursor()
    
    print(v_id, '<<<<<<<<<<<<<<<<<<<<<<<')
    sql= "select * from videos where id=%s "

    values=(v_id,)
    cursor2.execute(sql, values)
    records = cursor2.fetchall()
    title='';url='';img=''
    for row in records:
        title=row[3]
        url=row[1]
        img=row[4]
    
    sql= "select * from comments where id=%s "

    values=(v_id,)
    cursor2.execute(sql, values)
    data = cursor2.fetchall()

    res=[]

    for row in data:
        res.append(row[3])

    from Graph2 import generate
    graph=generate(res)

    
    return render_template("historydeatils.html",id=v_id, graph=graph, data=data, title=title, img=img, url=url)
    



@app.route("/searchaction",methods =["GET", "POST"])
def searchaction():
    try:
        url = request.form.get("url")
        from youtube import get_video_data
        from cleantext import extract_pure_text

        data=get_video_data(url)

        title=data[0]
        title_=extract_pure_text(title)
        img=data[1]
        comments=data[2]

        from sentiment import get_predictions
        res=get_predictions(comments)
        res=list(res)
        print(res, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

       

        database = DBConnection.getConnection()
        import random
        id = random.randint(9999, 99999)
        email=session['uid']

        cursor = database.cursor()
        sql = "insert into videos values(%s,%s,%s,%s,%s)"
        print(title, title_, '<<<<<<<<<<<<<<<<<<')
        print(img, '<<<<<<<<<<<<<<<<<<')
        values = (id,url, email, title_,img)
        cursor.execute(sql, values)
        database.commit()
        cursor = database.cursor()


        for i in range(len(comments)):
            sql = "insert into comments(id, comment, sentiment) values(%s,%s,%s)"
            com=extract_pure_text(comments[i])
            values = (id,com, int(res[i]))
            cursor.execute(sql, values)
        sql="update comments set sentiment2='negative' where sentiment=0"
        cursor.execute(sql)
        sql="update comments set sentiment2='neutral' where sentiment=1"
        cursor.execute(sql)
        sql="update comments set sentiment2='positive' where sentiment=2"
        cursor.execute(sql)
        
        sql="delete from comments where comment=''"
        cursor.execute(sql)
        
        database.commit()

        return render_template("searchres.html",id=id,  data=comments, title=title, img=img, url=url)

    except Exception as e:
        print(e)
        return render_template("search.html", msg='Invalid Video ID')
       



@app.route('/sentiresults',methods =["GET", "POST"])
def sentiresults():
    v_id = request.form.get("id")
    database = DBConnection.getConnection()
    cursor2 = database.cursor()
    
    print(v_id, '<<<<<<<<<<<<<<<<<<<<<<<')
    sql= "select * from videos where id=%s "

    values=(v_id,)
    cursor2.execute(sql, values)
    records = cursor2.fetchall()
    title='';url='';img=''
    for row in records:
        title=row[3]
        url=row[1]
        img=row[4]
    
    sql= "select * from comments where id=%s "

    values=(v_id,)
    cursor2.execute(sql, values)
    data = cursor2.fetchall()

    res=[]

    for row in data:
        res.append(row[3])

    from Graph2 import generate
    graph=generate(res)

    
    return render_template("sentiresults.html",id=v_id, graph=graph, data=data, title=title, img=img, url=url)
    


    

if __name__ == '__main__':
    app.run(host="localhost", port=1234, debug=False)
