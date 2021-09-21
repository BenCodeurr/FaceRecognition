import json,time
from camera import VideoCamera
from flask import Flask, render_template, request, jsonify, Response,redirect, url_for, flash
import requests
from flask_mysqldb import MySQL
import base64,cv2
import yaml
import train


app=Flask(__name__)
output=[]
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)

@app.route('/')
def home_page():
    return render_template("index.html",result=output)

@app.route('/login')
def login_page():
    return render_template("login.html", result=output)

@app.route('/enrolement', methods=['POST', 'GET'])
def enrolement_page():
    if request.method == 'POST':
        adminDetail = request.form
        nom = adminDetail['nomE']
        postnom = adminDetail['postnomE']
        prenom = adminDetail['prenomE']
        sexe= adminDetail['sexe']
        date = adminDetail['dateNaissance']
        nomPere = adminDetail['nomPere']
        nomMere = adminDetail['nomMere']
        phonePere = adminDetail['numPere']
        phoneMere = adminDetail['numMere']
        ville = adminDetail['ville']
        commune = adminDetail['commune']
        quartier = adminDetail['quartier']
        cellule = adminDetail['cellule']
        avenue = adminDetail['avenue']
        numParcelle = adminDetail['numParcelle']
        longitude = adminDetail['longitude']
        latitude = adminDetail['latitude']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO enfant (nomEnfant, postnomEnfant, prenomEnfant, sexe, nomPere, nomMere, dateNaissance, phonePere, phoneMere, ville, commune, quartier, cellule, avenue, numParcelle, longitude, latitude, idAdmin) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (nom, postnom, prenom, sexe, nomPere, nomMere, date, phonePere, phoneMere, ville, commune, quartier, cellule, avenue, numParcelle, longitude, latitude, 1))

        mysql.connection.commit()
        cur.close()
        return "<h1>reussi</h1>"
    
    return render_template("enrolement.html", result=output)


@app.route('/addAdmin', methods=['GET', 'POST'])
def addAdmin_page():
    if request.method == 'POST':
        adminDetail = request.form
        nom = adminDetail['nomAdmin']
        postnom = adminDetail['postnomAdmin']
        typeAdmin = adminDetail['typeAdmin']
        username = adminDetail['username']
        password = adminDetail['password']
        
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO admin (nomAdmin, postnomAdmin, username, password, typeAdmin) VALUES (%s, %s, %s, %s, %s)", (nom, postnom, username, password, typeAdmin))
        
        mysql.connection.commit()
        cur.close()
    return render_template("addAdmin.html", result=output)

@app.route('/dash')
def dash_page():
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM enfant')
    data = cur.fetchall()
    return render_template("dashboard.html", result=output, value=data)

@app.route('/train')
def train():
    print("Training KNN classifier...")
    classifier = train("dataSet/train", model_save_path="model/trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
    return classifier

def gen(camera):
    while True:
        data= camera.get_frame()

        frame=data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/recognizing')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)