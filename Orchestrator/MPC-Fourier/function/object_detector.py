
import numpy as np
import math
import cv2
#from tflite_runtime.interpreter import Interpreter, load_delegate
import tensorflow as tf



def map_camera_to_robot(camera_point, homograph_matrix):
    # Convert the point to homogeneous coordinates
    camera_point_homogeneous = np.append(camera_point, 1)

    # Apply the homography transformation
    robot_point_homogeneous = np.dot(homograph_matrix, camera_point_homogeneous)

    # Convert back to Cartesian coordinates
    robot_point = robot_point_homogeneous / robot_point_homogeneous[2]

    return robot_point[:2]
def preprocess_image(image_path, input_size):
    """Preprocess the input image using OpenCV for TFLite model."""
    img = cv2.imread(image_path)  # Read the image using OpenCV
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized_img = cv2.resize(img_rgb, input_size)  # Resize to match model input
    resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32)  # Add batch dimension
    return resized_img, img_rgb  # Return resized + original image

def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  # Copying the tensor data to avoid holding references to interpreter's internal data
  tensor = np.copy(interpreter.get_tensor(output_details['index']))
  return np.squeeze(tensor)


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results

def calculate_rotation_angle(x, y, xmin, ymin, xmax, ymax):
    # Calculate the center of the bounding box
    bbox_center_x = (xmin + xmax) // 2
    bbox_center_y = (ymin + ymax) // 2

    # Calculate the vector from the center of the image to the center of the bounding box
    dx = bbox_center_x - x
    dy = bbox_center_y - y

    # Calculate the angle (in radians) between the vector and the x-axis
    angle_rad = math.atan2(dy, dx)

    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg
def detect_objects_from_image(interpreter, input_details, classes,  img_np, threshold=0.5):


    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = input_details[0]['shape']

    # Resize the image to the required input size
    resized_img = cv2.resize(img_np, (input_width, input_height))
    resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32)
    # Add a batch dimension
    #resized_img = resized_img[tf.newaxis, :]

    #preprocessed_image = cv2.resize(img_tf, (input_width, input_height))
    #preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    # Run object detection on the input image
    results = detect_objects(interpreter, resized_img, threshold=threshold)

    # Process results
    object_locations = []
    object_angles = []
    list_coordinates = []
    for obj in results:
        if obj['score'] < 0.5:
            continue

        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin, xmax = int(xmin * img_np.shape[1]), int(xmax * img_np.shape[1])
        ymin, ymax = int(ymin * img_np.shape[0]), int(ymax * img_np.shape[0])

        # Calculate the center coordinates (x, y)
        x = (xmin + xmax) // 2
        y = (ymin + ymax) // 2
        object_locations.append((x, y))

        # Calculate angle of object
        x_img_center, y_img_center = img_np.shape[1] // 2, img_np.shape[0] // 2
        angle_deg = calculate_rotation_angle(x_img_center, y_img_center, xmin, ymin, xmax, ymax)
        #object_angles.append(angle_deg)

        # Find the class index of the current object
        class_id = int(obj['class_id'])
        obj_label = classes[class_id]
        # Convert all data types to avoid reference errors
        obj_info = {
            "obj": str(obj_label),
            "x_min": int(xmin),
            "x_max": int(xmax),
            "y_min": int(ymin),
            "y_max": int(ymax),
            "angle": float(angle_deg)
        }
        list_coordinates.append(obj_info)


    return list_coordinates