from flask import Flask, render_template, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("web.html")

@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    # Run your main.py (the TFT model)
    subprocess.run([sys.executable,"main.py"])
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)