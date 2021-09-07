import os
import PIL
from PIL import Image, ImageDraw, ImageOps

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

def image_handler(image, fixed_size = 1920):
    image = ImageOps.autocontrast(image.convert('L'), cutoff = 5, ignore = 5)
    w, h = image.size
    percent = float(fixed_size / float(min(w, h)))
    image = image.resize(
            (int(w * percent), int(h * percent)),
            PIL.Image.NEAREST
        )
    image.convert('RGB')
    return image

def resizer(file_path, file_name, save_path):
    image = Image.open(file_path)
    image = image_handler(image)
    image.save(save_path + file_name + '.jpg', optimize=True, quality=100)

def create_dataset(path_dataset, path_save):
    try:
        os.mkdir(path_save)
    except:
        pass
    folders = os.listdir(path_dataset)

    for folder in folders:
        os.mkdir(path_save + '/' + folder)
        images = [file for file in os.listdir(path_dataset + folder) if file.endswith(('jpeg', 'png', 'jpg'))]
        for i in images:
            filename = i.split('.')[0]
            resizer(
                file_path=path_dataset+folder+'/'+i,
                file_name=filename,
                save_path=path_save + '/' + folder + '/'
            )
