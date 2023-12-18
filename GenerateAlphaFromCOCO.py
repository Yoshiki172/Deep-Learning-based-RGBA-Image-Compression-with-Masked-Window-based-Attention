from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import os
import tqdm

# COCOデータセットのアノテーションファイルと画像ディレクトリのパス
annotation_file = './COCO/fast-ai-coco/annotations_trainval2017/annotations/instances_train2017.json'
image_dir = './COCO/fast-ai-coco/train2017/train2017/'

# RGBA画像を保存するディレクトリ
rgba_image_dir = 'P3Mdata/COCOdata'
os.makedirs(rgba_image_dir, exist_ok=True)

# COCOデータセットを読み込む
coco = COCO(annotation_file)

# カテゴリーIDと画像IDを取得
cat_ids = coco.getCatIds()
img_ids = coco.getImgIds()

# 画像ごとに処理
for img_id in tqdm.tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    rgb_img = Image.open(img_path).convert('RGB')

    # 新しい1チャンネルの画像を作成（初期値を0に設定）
    alpha_img = Image.new('L', (img_info['width'], img_info['height']), 0)
    draw = ImageDraw.Draw(alpha_img)
    
    # アノテーションを取得
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)
    
    for ann in anns:
        if 'segmentation' in ann and type(ann['segmentation']) == list and len(ann['segmentation']) > 0:
            segmentation = ann['segmentation'][0]
            
            # セグメンテーション領域を255（不透明）で塗りつぶす
            draw.polygon(segmentation, fill=255)
    
    # RGB画像とアルファチャンネルを結合
    rgba_img = Image.merge('RGBA', [rgb_img.split()[0], rgb_img.split()[1], rgb_img.split()[2], alpha_img])

    # RGBA画像を保存
    rgba_path = os.path.join(rgba_image_dir, img_info['file_name'].replace('.jpg', '.png'))
    rgba_img.save(rgba_path, 'PNG')