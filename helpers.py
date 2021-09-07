import PIL
from PIL import Image, ImageDraw

def draw_bboxs(image, preds_list):
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    for box in preds_list:
        coord = (tuple(box))
        draw.line((coord[0], coord[1], coord[0], coord[3]), fill=(255, 0, 0), width=5)
        draw.line((coord[0], coord[3], coord[2], coord[3]), fill=(255, 0, 0), width=5)
        draw.line((coord[2], coord[3], coord[2], coord[1]), fill=(255, 0, 0), width=5)
        draw.line((coord[2], coord[1], coord[0], coord[1]), fill=(255, 0, 0), width=5)
    return image

def resizer(image, fixed_size=1280, grayscale=True):
    w, h = image.size
    percent = float(fixed_size / float(max(w, h)))
    resized_image = image.resize(
            (int(w * percent), int(h * percent)),
            PIL.Image.NEAREST
        )
    if grayscale:
        resized_image = resized_image.convert('L') 
    return resized_image_gray
