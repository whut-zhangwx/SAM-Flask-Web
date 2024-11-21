"""server.py"""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/receive", methods=["POST"])
def receive():
  print(request.headers)
  
  files = request.files
  file = request.files["file"]
  
  # json = request.json
  # print(f"json: {json},\ntype(json): {type(json)}")
  # json: {'name': 'json', 'content': 456},
  # type(json): <class 'dict'>

  print(f"files: {files},\ntype(files): {type(files)}")
  # files: ImmutableMultiDict([('file', <FileStorage: 'image.jpg' (None)>)]),
  # type(files): <class 'werkzeug.datastructures.structures.ImmutableMultiDict'>
  print(f"files.name: {files.keys()}")

  
  print(f"file: {file},\n"
        f"type(file): {type(file)},\n"
        f"file.name: {file.name},\n"
        f"file.filename: {file.filename},\n"
        f"file.content_type: {file.content_type}")
        
  # file: <FileStorage: 'image.jpg' (None)>, file.name: file, file.filename: image.jpg,
  # type(file): <class 'werkzeug.datastructures.file_storage.FileStorage'>

  return jsonify({"success": True})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)