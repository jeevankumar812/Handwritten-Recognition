from flask import Flask,render_template,request
import os
from predict import predict_word

app = Flask(__name__)

UPLOAD = "static/uploads"

# ✅ AUTO CREATE FOLDER
os.makedirs(UPLOAD, exist_ok=True)

@app.route("/",methods=["GET","POST"])
def home():
    result=""
    if request.method=="POST":
        f=request.files["image"]
        path=os.path.join(UPLOAD,f.filename)
        f.save(path)
        result=predict_word(path)
    return render_template("index.html",result=result)

if __name__=="__main__":
    app.run(debug=True)
