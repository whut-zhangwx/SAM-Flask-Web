from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import base64

import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

device = "cuda"
model_type, sam_checkpoint = "vit_b", "../assets/sam_vit_b_01ec64.pth"

class SamFlask(Flask):
   def __init__(self, import_name: str, model_type: str, sam_ckeckpoint: str, device: str = 'cpu') -> None:
      super().__init__(import_name=import_name)
      ''' load the SAM model and predictor.'''
      self.sam = sam_model_registry[model_type](checkpoint=sam_ckeckpoint)
      self.sam.to(device)
      self.predictor = SamPredictor(self.sam)
      self.image = None
      self.image_bgra = None
      logging.info(f"SAM Initialization is Complete")

# convert a 1-channel zero-one mask to a bgra 4-channel image
def mask2bgra(mask: np.ndarray) -> np.ndarray:
  '''
  mask.shape: [1, H, W] | [H, W]
  BGRA, A(Alpha)-Transparency: 0-Completely transparent, 255-Completely opaque
  '''
  color = np.array([255, 144, 30, 100], dtype=np.uint8) # BGRA
  h, w = mask.shape[-2:]
  '''
  masks.reshape(h, w, 1): [H, W, 1], color.reshape(1, 1, -1): [1, 1, 4]
  [H, W, 1] * [1, 1, 4] -> [H, W, 4]
  '''
  mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
  return mask_image

# convert a bgr 3-channel image to a bgra 4-channel image
def bgr2bgra(image_bgr: np.ndarray) -> np.ndarray:
  '''image_bgr.shape: [H, W, 3]'''
  logging.info(f"image.shape: {image_bgr.shape}, image.dtype: {image_bgr.dtype}")
  if(image_bgr.shape[2] == 4): return image_bgr
  h, w = image_bgr.shape[:2]
  alpha_channnel = np.ones(shape=(h, w, 1), dtype=np.uint8) * 255 # 255 means 100% opaque
  image_bgra = cv2.merge((image_bgr, alpha_channnel)) # BGR + A -> BGRA
  return image_bgra

# calculate the area proportion of segmented area in whole image
def proportion_of_area(mask: np.ndarray) -> np.float64:
  '''mask.shape: [1, H, W] | [H, W]'''
  return np.sum(mask, axis=None) / np.multiply(*mask.shape)


app = SamFlask(__name__, model_type, sam_checkpoint, device)

'''Setting Access-Control-Allow-Origin'''
CORS(app) # Allow all ip to access
# Only allow specific ip to access
# CORS(app, origins=["http://localhost:5501/"])


@app.route("/predictor_set_image", methods=["POST"])
def predictor_set_image():
  '''load the image from front end and process the image to produce an image embedding by calling SamPredictor.set_image'''
  logging.info("predictor_set_image() is called")

  try:
    '''load the image from front end'''
    file = request.files['image']
    logging.info(f"type(file): {type(file)}") # type(file): <class 'werkzeug.datastructures.file_storage.FileStorage'>
    file_bytes = file.read()
    logging.info(f"type(file_bytes): {type(file_bytes)}") # type(file_bytes): <class 'bytes'>
    image = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # BGR
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB (don't do this, it will cause confusion)
    app.image = image
    logging.info(f"image.shape: {image.shape}, type: {type(image)}")
    app.image_bgra = bgr2bgra(image)
    logging.info(f"image_bgra.shape: {app.image_bgra.shape}, type: {type(app.image_bgra)}")
    '''
    Process the image to produce an image embedding by calling SamPredictor.set_image.
    SamPredictor remembers this embedding and will use it for subsequent mask prediction.
    '''
    app.predictor.set_image(image)
    return jsonify({'success': True})
  except:
    return jsonify({'success': False})


@app.route("/decode_embedding", methods=["POST"])
def decode_embedding():
  logging.info("decode_embedding() is called")

  points = json.loads(request.data)['points']
  logging.info(f"points: {points}, type: {type(points)}")

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

  '''
  Predict with SamPredictor.predict. The model returns masks, quality predictions for those masks,
  and low resolution mask logits that can be passed to the next iteration of prediction.
  '''
  masks, scores, logits = app.predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
  )
  '''
  With `multimask_output=True` (the default setting), SAM outputs 3 masks, where scores gives the model's own estimation of the quality of these masks.
  This setting is intended for ambiguous input prompts, and helps the model disambiguate different objects consistent with the prompt.
  When False, it will return a single mask. For ambiguous prompts such as a single point, it is recommended to use multimask_output=True even if only a single mask is desired;
  the best single mask can be chosen by picking the one with the highest score returned in scores. This will often result in a better mask.
  '''
  logging.info(f"masks.shape: {masks.shape}, scores.shape: {scores.shape}")

  mask = masks[np.argmax(scores)] # get the mask with the highest score of the three masks
  proportion = proportion_of_area(mask) # get the proporation of the segmented area in whole image

  mask_image = mask2bgra(mask=mask) # convet the 0-1 single channel mask to BGRA 4 channel image
  logging.info(f"mask_image.shape: {mask_image.shape}, type: {type(mask_image)}")

  '''Convert BGR Image to BGRA Image'''
  # image_bgra = bgr2bgra(image_bgr=app.image) # BGR + A -> BGRA
  # logging.info(f"image_bgra.shape: {image_bgra.shape}, type: {type(image_bgra)}")

  '''Blend The Two BGRA Images with weights of 1 : 1'''
  blended_image = cv2.addWeighted(app.image_bgra, 1, mask_image, 1, 0)
  logging.info(f"blended_image.shape: {blended_image.shape}, type: {type(blended_image)}")

  '''encode the image to png'''
  ret, png_image = cv2.imencode('.png', blended_image)
  if not ret: return jsonify({"status": "Failed", "message": "Failed to encode image"})
  base64_image = base64.b64encode(png_image).decode('utf-8')

  return jsonify({"image": base64_image, "proportion": proportion})


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)

# python flask_server_pytorch.py
# http://0.0.0.0:5000/