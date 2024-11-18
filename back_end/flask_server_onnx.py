from flask import Flask, request, Request, Response, jsonify, send_file
from flask_cors import CORS # 
import json
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime
import logging
from io import BytesIO
import base64

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

device = "cuda"

model_type, path2ckpt = "vit_b", "../assets/sam_vit_b_01ec64.pth"
sam = sam_model_registry[model_type](checkpoint=path2ckpt)
sam.to(device=device)
predictor = SamPredictor(sam)

onnx_model_path = "../assets/sam_onnx_vit_b.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

image = None
image_embedding = None

app = Flask(__name__)
'''Setting Access-Control-Allow-Origin'''
CORS(app) # Allow all ip to access
# Only allow specific ip to access
# CORS(app, origins=["http://localhost:5501/"])

@app.route('/upload_image', methods=['POST'])
def upload_image():
    logging.info("upload_image() is called")
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    file = request.files['image']
    file.save(f"./tempFiles/{file.filename}")
    return jsonify({'success': True, 'message': 'Image uploaded successfully'})


@app.route("/embed_image", methods=["POST"])
def embed_image():
  logging.info("embed_image() is called")

  global image, image_embedding

  try:
    # image = request.data
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    # image = cv2.imdecode(image, 1)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # BGR
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
    logging.info(f"image.shape: {image.shape}, type: {type(image)}")
    # height, width, channel = image.shape
    # print(image[0][0])
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    logging.info(f"image_embedding.shape: {image_embedding.shape}")
    return jsonify({'status': 'Success'})
    # height, width, channel = image.shape
    # return jsonify({'status': 'Success', 'imageHeight': height, 'imageWidth': width})
    # predictor.set_image(image)
    # image_embedding = predictor.get_image_embedding().cpu().numpy()
  except:
    return jsonify({'status': 'Failed'})
  # return jsonify({'status': 'Success'})


@app.route("/decode_embedding", methods=["POST"])
def decode_embedding():
  logging.info("decode_embedding() is called")
  points = json.loads(request.data)['points']
  logging.info(f"points: {points}, type: {type(points)}")

  global image, image_embedding

  '''
  the point (x, y) that will be passed to mask decoder
  +——————————+
  |    | y   |
  +----+     | height
  |  x       |
  +——————————+
      width
  '''
  input_point = np.array(points)
  input_label = np.array([1 for _ in range(len(points))])
  logging.info(f"input_point: {input_point}, type: {type(input_label)}")

  # Add a batch index, concatenate a padding point, and transform.
  onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
  onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
  onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

  # Create an empty mask input and an indicator for no mask.
  onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
  onnx_has_mask_input = np.zeros(1, dtype=np.float32)

  # Package the inputs to run in the onnx model
  ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
  }

  masks, _, low_res_logits = ort_session.run(None, ort_inputs)
  masks = masks > predictor.model.mask_threshold
  # mask.shape: [B, C, H, W]
  logging.info(f"masks.shape: {masks.shape}, type: {type(masks)}")

  # color = B, G, R, A
  color = np.array([255, 144, 30, 100], dtype=np.uint8) # A(Alpha)-Transparency: 0-Completely transparent, 255-Completely opaque
  h, w = masks.shape[-2:]
  # masks.reshape(h, w, 1): [H, W, 1], color.reshape(1, 1, -1): [1, 1, 4]
  mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
  logging.info(f"mask_image.shape: {mask_image.shape}, type: {type(mask_image)}")
  # cv2.imwrite("mask_image.png", mask_image)

  '''Convert BGR Image to BGRA Image'''
  # height, width = image.shape[:2]
  alpha_channnel = np.ones(shape=(h, w), dtype=np.uint8) * 255
  image_bgra = cv2.merge((image, alpha_channnel)) # RGB + A -> RGBA
  logging.info(f"image_rgba.shape: {image_bgra.shape}, type: {type(image_bgra)}")

  '''Blend The Two RGBA Images'''
  blended_image = cv2.addWeighted(image_bgra, 1, mask_image, 1, 0)

  logging.info(f"blended_image.shape: {blended_image.shape}, type: {type(blended_image)}")

  ret, png_image = cv2.imencode('.png', blended_image)
  if not ret: return jsonify({"status": "Failed", "message": "Failed to encode image"})

  base64_image = base64.b64encode(png_image).decode('utf-8')

  return jsonify({"image": base64_image})

  # # 将 JPEG 字节数据转为流格式以便传输
  # image_bytes = BytesIO(jpeg_image.tobytes())
  # image_bytes.seek(0)

  # return send_file(image_bytes, mimetype='image/jpeg')

  # return jsonify({'status': 'Success', 'masks': mask_image})
  # return masks.tobytes()


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)

# python flask_server.py