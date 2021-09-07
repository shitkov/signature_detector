import PIL
from PIL import Image, ImageDraw

def draw_bboxs(img_path, preds_list):
	image = Image.open(img_path)
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    for box in preds_list:
        coord = (tuple(box))
        draw.line((coord[0], coord[1], coord[0], coord[3]), fill=(255, 0, 0), width=5)
        draw.line((coord[0], coord[3], coord[2], coord[3]), fill=(255, 0, 0), width=5)
        draw.line((coord[2], coord[3], coord[2], coord[1]), fill=(255, 0, 0), width=5)
        draw.line((coord[2], coord[1], coord[0], coord[1]), fill=(255, 0, 0), width=5)
    return image
