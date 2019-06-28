import os
from  object_tracker_test import predict
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request
from sqlalchemy.sql import func
from apscheduler.schedulers.background import BackgroundScheduler

def scheduked_task():
    rows = Example.query.filter_by(processed = 0,processing = 0)
    # print(len(rows))
    for i in rows:
        print("fr3")
        file_name = i.file_name
        print(file_name)
        id = i.id
        updated_row = Example.query.filter_by(id = id).first()
        updated_row.processing = 1
        db.session.commit()
    for i in rows:
        updated_row = Example.query.filter_by(id = id).first()
        l = predict(file_name)
        updated_row.processed = 1
        updated_row.processing = 0
        updated_row.total_persons = l[0]
        updated_row.male = l[1]
        updated_row.female = l[2]
        db.session.commit()

sched = BackgroundScheduler(daemon=True)
sched.add_job(scheduked_task,'interval',minutes=1)
sched.start()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/ml_project_test'
db = SQLAlchemy(app)

class Example(db.Model):
    __tablename__ = 'sample_test3'
    id = db.Column('id',db.Integer, primary_key=True)
    file_name = db.Column('filename',db.Unicode)
    measured_on = db.Column('measured_on',db.TIMESTAMP)
    processed = db.Column('processed',db.Integer,default=0)
    processing = db.Column('processing',db.Integer,default=0)
    total_persons = db.Column('total_persons',db.Integer)
    male = db.Column('male',db.Integer)
    female = db.Column('female',db.Integer)

    def __init__(self, file_name):
        # self.id = id
        self.file_name = file_name
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/upload" ,methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT)
    print(target)

    # if not os.path.isdir(target):
    #     os.mkdir(target)
    for file in request.files.getlist("input_file"):
        # print(file)
        filename = file.filename
        print(filename)
        destination = "/".join([target, filename])
        # print(destination)
        file.save(destination)
        new_ex = Example(filename)
        db.session.add(new_ex)
        db.session.commit()
        # l = predict(filename)
    # return 'Total Number of persons detected %d' % l[0]
    return 'Video Uploaded Successfully.'

@app.route("/processed_videos")
def processed_videos():
    processed_rows = Example.query.filter_by(processed = 1)
    return render_template("processed_videos.html",data = processed_rows)

if __name__=="__main__":
    app.run(port=4555, DEBUG =True)
