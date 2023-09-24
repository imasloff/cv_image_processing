import os
import cv2
import json
import numpy as np

INPUT_DIR = "dataset"
OUTPUT_DIR = "output"

def extract_lower_body_coordinates(pose_data):
    """
    Extracts lower body points
    """
    pose_keypoints = pose_data['people'][0]['pose_keypoints_2d']
    l_hand_keypoints = pose_data['people'][0]['hand_left_keypoints_2d']
    r_hand_keypoints = pose_data['people'][0]['hand_right_keypoints_2d']
    hands_keypoints = l_hand_keypoints + r_hand_keypoints
    pose_points = [(int(pose_keypoints[i]), int(pose_keypoints[i + 1])) for i in range(0, len(pose_keypoints), 3)]
    hands_points = [(int(hands_keypoints[i]), int(hands_keypoints[i + 1])) for i in range(0, len(hands_keypoints), 3)]
    return list(set(pose_points[8:11] + pose_points[12:14]) - set(hands_points))

def shade_lower_body(image, lower_body_coordinates, human_parse):
    """
    Generates shaded image
    """
    # making color mask using human parse and lower body coordinates
    colors = [human_parse[y][x] for x, y in lower_body_coordinates]
    color_mask = np.zeros((human_parse.shape[0], human_parse.shape[1]), dtype=bool)
    for color in colors:
        mask = np.all(human_parse == color, axis=-1)
        color_mask = color_mask | mask
    
    # making lower body mask from corresponding regions in human parse
    lower_body_mask = np.zeros_like(human_parse, np.uint8)
    lower_body_mask[color_mask] = human_parse[color_mask]
    coordinates = np.argwhere(np.any(lower_body_mask != 0, axis=2)).astype(np.int32)
    coordinates = coordinates[:, [1, 0]]

    # filling lower body mask with color and putting it onto image
    lower_body_mask = cv2.fillConvexPoly(lower_body_mask, coordinates, (132,134,136))
    return cv2.addWeighted(image, 1, lower_body_mask, 5, 0)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(os.path.join(INPUT_DIR, "image")):
        # reading the image
        image_path = os.path.join(INPUT_DIR, "image", filename)
        image_id = filename.split('.')[0]
        image = cv2.imread(image_path)

        # reading the corresponding human parse
        human_parsing_path = os.path.join(INPUT_DIR, "human_parsing", f"{image_id}.png")
        human_parse = cv2.imread(human_parsing_path)

        # reading the corresponding json
        pose_json_path = os.path.join(INPUT_DIR, "pose_json", f"{image_id}_keypoints.json")
        with open(pose_json_path, 'r') as pose_json_file:
            pose_data = json.load(pose_json_file)

        # getting the shaded image
        shaded_image = shade_lower_body(image, extract_lower_body_coordinates(pose_data), human_parse)

        # saving the shaded image to the output folder
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'shaded_{filename}'), shaded_image)
