import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry


def main():
    model_type = 'vit_h'
    checkpoint_path = r'D:\Git\segment-anything\models\sam_vit_h_4b8939.pth'
    p_img = r'\\192.168.0.225\algo_group\FaceAttributes\FaceMask\Positive\Polygon-iteration4\images\driving_videos\Dana_Driving_28APR2020_Video9_3.png'

    assert os.path.isfile(p_img)
    assert os.path.isfile(checkpoint_path)

    img = cv2.imread(p_img, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    pt_prompt = np.array([(465, 570)])

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

    t = time.time()
    predictor.set_image(img)
    elapsed = time.time() - t
    print(elapsed)

    t = time.time()
    masks, _, _ = predictor.predict(point_coords=pt_prompt, point_labels=np.array([(1)]))
    elapsed = time.time() - t
    print(elapsed)

    m = 255 * np.transpose(masks, (1, 2, 0)).astype(np.uint8)
    plt.imshow(m)
    plt.show()
    a = 1


if __name__ == '__main__':
    main()
