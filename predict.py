import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata
from torchvision.datasets.utils import download_url

# 画像の取得
url = 'https://github.com/dai-ichiro/robo-one/raw/main/test.jpg'
fname = url.split('/')[-1]
download_url(url, root = '.', filename = fname)
im = cv2.imread(fname)

# モデルの取得
model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
cfg = model_zoo.get_config(model, trained=False)
cfg.MODEL.WEIGHTS = 'output/model_final.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  

# メタデータの新規作成
original_metadata = Metadata()
original_metadata.thing_classes = ['target', 'green']

predictor = DefaultPredictor(cfg)

outputs = predictor(im)

v = Visualizer(im, original_metadata, scale=1.0)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
img_array = v.get_image()

cv2.imshow ('result', img_array)
cv2.waitKey(0)
cv2.destroyAllWindows()