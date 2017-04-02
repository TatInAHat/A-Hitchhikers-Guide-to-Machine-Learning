from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
	return redirect("http://cool.us-west-2.elasticbeanstalk.com/index.html", code=302)

if __name__ == "__main__":
app.run()