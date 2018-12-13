#!/usr/bin/python

__ATTR__ = {
    'CelebA': [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ],
    'RafD': [
        'neutral', 'angry', 'contemptuous', 'disgusted', 'fearful', 'happy',
        'sad', 'surprised'
    ],
    'painters_14': [
        'beksinski', 'boudin', 'burliuk', 'cezanne', 'chagall', 'corot',
        'earle', 'gauguin', 'hassam', 'levitan', 'monet', 'picasso', 'ukiyoe',
        'vangogh'
    ],
    'Animals': [
        'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian',
        'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat',
        'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose',
        'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox',
        'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros',
        'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel',
        'otter', 'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', 'pig',
        'lion', 'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow',
        'dolphin'
    ],
    'Image2Season': ['autumn', 'spring', 'summer', 'winter']
}


def replace_break_line(text):
    if '_o_' in text:
        text = text.replace('_o_', '_o ')
    if '_' in text:
        text = text.replace('_', '\n')
    if '+' in text:
        text = text.replace('+', '\n')
    text = text.split('\n')
    return text


def get_max_size(FONT, text, base_size):
    max_size = 70
    font = FONT(max_size)
    while (font.getsize(text)[0] >= base_size):
        max_size -= 1
        font = FONT(max_size)
    return max_size


def get_font():
    from PIL import ImageFont
    return lambda size: ImageFont.truetype("data/Times-Roman.otf", size)


def get_img(text, background='white', size=None):
    from PIL import ImageDraw, Image
    base_size = (256, 256)
    foreground = (0, 0, 0) if background == 'white' else (255, 255, 255)
    background = (255, 255, 255) if background == 'white' else (0, 0, 0)
    img = Image.new('RGB', base_size, background)

    text = replace_break_line(text)
    text = [line.capitalize() for line in text]

    draw = ImageDraw.Draw(img)
    FONT = get_font()
    if size is None:
        size = []
        for idx, _text in enumerate(text):
            size.append(get_max_size(FONT, _text, base_size[0]))
        size = min(size)

    font = FONT(size)
    previous_y = 0
    for _text in text[::-1]:
        _text = _text
        textsize = font.getsize(_text)
        textX = (img.size[0] - textsize[0]) / 2
        textY = img.size[1] - textsize[1] - previous_y - 5
        draw.text((textX, textY), _text, font=font, fill=foreground)
        previous_y += textsize[1]
    return img


def external2img(attributes, img_size):
    FONT = get_font()
    size = []
    for idx, attr in enumerate(attributes):
        text = replace_break_line(attr)
        text = [line.capitalize() for line in text]
        for _text in text:
            size.append(get_max_size(FONT, _text, img_size))
    size = min(size)
    return text2img(attributes, size=size)


def text2img(attributes, save=None, size=None):

    assert isinstance(attributes, list)
    imgs = []
    for idx, attr in enumerate(attributes):
        color = 'white'  # 'black' if attr in ['Source', 'Off'] else 'white'
        text = attr.capitalize()  # if dataset!= 'Birds' else attr
        img = get_img(text, color, size=size)
        imgs.append(img)
        if save is not None:
            img.save(save)
    if save is None:
        return imgs


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all')
    opt = parser.parse_args()
    if opt.dataset != 'all':
        assert os.path.isdir(opt.dataset)
        opt.dataset = [opt.dataset]
    else:
        opt.dataset = __ATTR__.keys()

    for dataset in opt.dataset:
        folder = os.path.join(dataset, 'aus_flat')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        __ATTR__[dataset] = ['Source', 'Off'] + __ATTR__[dataset]
        text2img(__ATTR__[dataset], save=folder)
