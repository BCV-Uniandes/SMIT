from __future__ import print_function


# ==================================================================#
# ==================================================================#
def circle_frame(tensor, thick=5, color='green', row_color=None):
    import numpy as np
    import torch
    from scipy import ndimage
    _color_ = {'green': (-1, -1, 1), 'red': (-1, -1, 1), 'blue': (-1, -1, 1)}
    _tensor = tensor.clone()
    size = tensor.size(2)
    thick = ((size / 2)**2) / 7.5
    xx, yy = np.mgrid[:size, :size]
    circle = (xx - size / 2)**2 + (yy - size / 2)**2
    donut = np.logical_and(circle < ((size / 2)**2 + thick),
                           circle > ((size / 2)**2 - thick))
    if color == 'blue':
        donut = ndimage.binary_erosion(
            donut, structure=np.ones((15, 1))).astype(donut.dtype)
    elif color == 'red':
        donut = ndimage.binary_erosion(
            donut, structure=np.ones((1, 15))).astype(donut.dtype)
    donut = np.expand_dims(donut, 0) * 1.
    donut = donut.repeat(tensor.size(1), axis=0)
    for i in range(donut.shape[0]):
        donut[i] = donut[i] * _color_[color][i]
    donut = to_var(torch.FloatTensor(donut), volatile=True)
    if row_color is None:
        row_color = [0, -1]
    else:
        row_color = [row_color]
    for nn in row_color:  # First and last frame
        _tensor[nn] = tensor[nn] + donut
    return _tensor


# ==================================================================#
# ==================================================================#
def compute_lpips(img0, img1, model=None):
    # RGB image from must be [-1,1]
    if model is None:
        from lpips_model import DistModel
        model = DistModel()
        version = '0.0'  # Totally different values with 0.1
        model.initialize(
            model='net-lin', net='alex', use_gpu=True, version=version)
    dist = model.forward(img0, img1)
    return dist, model


# ==================================================================#
# ==================================================================#
def config_yaml(config, yaml_file):
    def dict_dataset(dict):
        import os
        if 'dataset' in dict.keys():
            config.dataset_fake = os.path.join(config.dataset_fake,
                                               dict['dataset'])

    import yaml
    with open(yaml_file, 'r') as stream:
        config_yaml = yaml.load(stream)
    dict_dataset(config_yaml)
    for key, value in config_yaml.items():
        if 'ALL_ATTR' in key and config.ALL_ATTR > 0:
            config.c_dim = config_yaml['ALL_ATTR_{}'.format(
                config.ALL_ATTR)]['c_dim']
            dict_dataset(config_yaml['ALL_ATTR_{}'.format(config.ALL_ATTR)])
        else:
            setattr(config, key, value)


# ==================================================================#
# ==================================================================#
def color_frame(tensor, thick=5, color='green', first=False):
    _color_ = {'green': (-1, 1, -1), 'red': (1, -1, -1), 'blue': (-1, -1, 1)}
    # tensor = to_data(tensor)
    for i in range(thick):
        for k in range(tensor.size(1)):
            # for nn in [0,-1]: #First and last frame
            for nn in [0]:  # First
                tensor[nn, k, i, :] = _color_[color][k]
                if first:
                    tensor[nn, k, :, i] = _color_[color][k]
                tensor[nn, k, tensor.size(2) - i - 1, :] = _color_[color][k]
                tensor[nn, k, :, tensor.size(2) - i - 1] = _color_[color][k]
    return tensor


# ==================================================================#
# ==================================================================#
def create_arrow(img_path, style, image_size=256, horizontal=False):

    if style == 0:
        text = 'Multimodality'
    elif style == 1:
        text = 'Multimodal Interp.'
    elif style == 2:
        text = 'Multi-label Interp.'
    else:
        return

    import cv2 as cv
    import numpy as np
    from PIL import Image, ImageFont, ImageDraw
    img = cv.imread(img_path)

    # Draw arrow
    size = image_size
    if horizontal:
        n_start = 1
    else:
        n_start = 2
    pointY = [(n_start * size) + size // 2, img.shape[0] - size // 2]
    pointX = [size // 2, size // 2]
    cv.arrowedLine(
        img, (pointX[0], pointY[0]), (pointX[1], pointY[1]), (0, 0, 0),
        int(size // 4),
        tipLength=0.08)
    cv.arrowedLine(
        img, (pointX[1], pointY[1]), (pointX[0], pointY[0]), (0, 0, 0),
        int(size // 4),
        tipLength=0.08)

    if horizontal:
        ones = np.ones(
            (size, img.shape[1], img.shape[2])).astype(np.uint8) * 255
        img = np.concatenate((img, ones), axis=0)
        pointY_h = [img.shape[0] - size // 2, img.shape[0] - size // 2]
        pointX_h = [size + size // 2, img.shape[1] - size // 2]
        cv.arrowedLine(
            img, (pointX_h[0], pointY_h[0]), (pointX_h[1], pointY_h[1]),
            (0, 0, 0),
            int(size // 4),
            tipLength=0.08)
        cv.arrowedLine(
            img, (pointX_h[1], pointY_h[1]), (pointX_h[0], pointY_h[0]),
            (0, 0, 0),
            int(size // 4),
            tipLength=0.08)

    cv.imwrite(img_path, img)
    # Write text
    font = ImageFont.truetype("data/Times-Roman.otf", int(size // 2.5) // 2)
    textsize = font.getsize(text)
    textX = size // 2 + (pointY[1] - pointY[0] - textsize[0]) // 2
    if horizontal:
        textX += size
    textY = size // 2 - textsize[1] // 2
    img = Image.open(img_path).convert('RGB').rotate(-90, expand=1)
    draw = ImageDraw.Draw(img)
    draw.text((textX, textY), text, font=font)
    img = img.rotate(90, expand=1)
    img.save(img_path)
    if horizontal:
        textX = size + size // 2 + (
            pointX_h[1] - pointX_h[0] - textsize[0]) // 2
        textY = img.size[1] - size // 2 - textsize[1] // 2
        draw = ImageDraw.Draw(img)
        draw.text((textX, textY), text, font=font)
        img.save(img_path)


# ==================================================================#
# ==================================================================#
def create_circle(size=256):
    import numpy as np
    xx, yy = np.mgrid[:size, :size]
    # circles contains the squared distance to the (size, size) point
    # we are just using the circle equation learnt at school
    circle = (xx - size / 2)**2 + (yy - size / 2)**2
    bin_circle = (circle <= (size / 2)**2) * 1.
    return bin_circle


# ==================================================================#
# ==================================================================#
def create_dir(dir):
    import os
    if '.' in os.path.basename(dir):
        dir = os.path.dirname(dir)
    if not os.path.isdir(dir):
        os.makedirs(dir)


# ==================================================================#
# ==================================================================#
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


# ==================================================================#
# ==================================================================#
def get_labels(image_size, dataset, attr=None):
    import imageio
    import glob
    import torch
    import skimage.transform
    import numpy as np

    def imread(x):
        return imageio.imread(x)

    def resize(x):
        return skimage.transform.resize(x, (image_size, image_size))

    if dataset not in ['EmotionNet', 'BP4D']:
        from data.attr2img import external2img
        selected_attrs = []
        for attr in attr.selected_attrs:
            if attr == 'Male':
                attr = 'Male/_Female'
            elif attr == 'Young':
                attr = 'Young/_Old'
            elif 'Hair' in attr:
                pass
            elif dataset == 'CelebA':
                attr += '_Swap'
            else:
                pass
            selected_attrs.append(attr)

        labels = ['Source'] + selected_attrs
        imgs = external2img(labels, img_size=image_size)
        imgs = [resize(np.array(img)).transpose(2, 0, 1) for img in imgs]
    else:
        imgs_file = sorted(glob.glob('data/{}/aus_flat/*g'.format(dataset)))
        imgs_file.pop(1)  # Removing 'off'
        imgs = [resize(imread(line)).transpose(2, 0, 1) for line in imgs_file]
    imgs = torch.from_numpy(np.concatenate(imgs, axis=2).astype(
        np.float32)).unsqueeze(0)
    return imgs


# ==================================================================#
# ==================================================================#
def get_loss_value(x):
    if get_torch_version() > 0.3:
        return x.item()
    else:
        return x.data[0]


# ==================================================================#
# ==================================================================#
def get_torch_version():
    import torch
    return float('.'.join(torch.__version__.split('.')[:2]))


# ==================================================================#
# ==================================================================#
def imgShow(img):
    from torchvision.utils import save_image
    try:
        save_image(denorm(img).cpu(), 'dummy.jpg')
    except BaseException:
        save_image(denorm(img.data).cpu(), 'dummy.jpg')


# ==================================================================#
# ==================================================================#
def load_inception(path='data/RafD/normal/inception_v3.pth'):
    from torchvision.models import inception_v3
    import torch
    import torch.nn as nn
    state_dict = torch.load(path)
    net = inception_v3(pretrained=False, transform_input=True)
    print("Loading inception_v3 from " + path)
    net.aux_logits = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, state_dict['fc.weight'].size(0))
    net.load_state_dict(state_dict)
    for param in net.parameters():
        param.requires_grad = False
    return net


# ==================================================================#
# ==================================================================#
def make_gif(imgs, path, im_size=256, total_styles=5):
    import imageio
    import numpy as np
    if 'jpg' in path:
        path = path.replace('jpg', 'gif')
    imgs = (imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    target_size = (im_size, im_size, imgs.shape[-1])
    img_list = []
    for x in range(imgs.shape[2] // im_size):
        for bs in range(imgs.shape[0]):
            if x == 0 and bs > 1:
                continue  # Only save one image of the originals
            if x == 1:
                continue  # Do not save any of the 'off' label
            img_short = imgs[bs, :, im_size * x:im_size * (x + 1)]
            assert img_short.shape == target_size
            img_list.append(img_short)
    imageio.mimsave(path, img_list, duration=0.8)

    writer = imageio.get_writer(path.replace('gif', 'mp4'), fps=3)
    for im in img_list:
        writer.append_data(im)
    writer.close()


# ==================================================================#
# ==================================================================#
def Modality(target, style, Multimodality):
    import numpy as np
    import torch

    # Style interpolation | Fixed Labels
    if Multimodality == 2:
        z0 = to_data(style[0], cpu=True).numpy()
        z1 = to_data(style[1], cpu=True).numpy()
        z_interp = style.clone()
        z_interp[:] = torch.FloatTensor(
            np.array([
                slerp(sz, z0, z1) for sz in np.linspace(0, 1, style.size(0))
            ]))
        style = z_interp

    # Style constant | Progressive swap label
    elif Multimodality == 3:
        label_space = np.linspace(0, 1, target.size(0))
        for j, i in enumerate(range(target.size(0))):
            style[i] = style[0].clone()
            target[i].data.fill_(
                (target[i] * label_space[j] +
                 (1 - target[i]) * (1 - label_space[j])).data[0])

    return target, style


# ==================================================================#
# ==================================================================#


def one_hot(labels, dim):
    """Convert label indices to one-hot vector"""
    import torch
    import numpy as np
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


# ==================================================================#
# ==================================================================#
def PRINT(file, str):
    print(str, file=file)
    file.flush()
    print(str)


# ==================================================================#
# ==================================================================#
def plot_txt(txt_file):
    import matplotlib.pyplot as plt
    lines = [line.strip().split() for line in open(txt_file).readlines()]
    legends = {idx: line
               for idx, line in enumerate(lines[0][1:])}  # 0 is epochs
    lines = lines[1:]
    epochs = []
    losses = {loss: [] for loss in legends.values()}
    for line in lines:
        epochs.append(line[0])
        for idx, loss in enumerate(line[1:]):
            losses[legends[idx]].append(float(loss))

    import pylab as pyl
    plot_file = txt_file.replace('.txt', '.pdf')
    _min = 4 if len(losses.keys()) > 9 else 3
    for idx, loss in enumerate(losses.keys()):
        # plot_file = txt_file.replace('.txt','_{}.jpg'.format(loss))

        plt.rcParams.update({'font.size': 10})
        ax1 = plt.subplot(3, _min, idx + 1)
        # err = plt.plot(epochs, losses[loss], 'r.-')
        err = plt.plot(epochs, losses[loss], 'b.-')
        plt.setp(err, linewidth=2.5)
        plt.ylabel(loss.capitalize(), fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        ax1.tick_params(labelsize=8)
        plt.hold(False)
        plt.grid()
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    pyl.savefig(plot_file, dpi=100)


# ==================================================================#
# ==================================================================#
def pdf2png(filename):
    from wand.image import Image
    from wand.color import Color
    import os
    with Image(filename="{}.pdf".format(filename), resolution=500) as img:
        with Image(
                width=img.width, height=img.height,
                background=Color("white")) as bg:
            bg.composite(img, 0, 0)
            bg.save(filename="{}.png".format(filename))
    os.remove('{}.pdf'.format(filename))


# ==================================================================#
# ==================================================================#
def replace_weights(target, source, list):
    for l in list:
        target[l] = source[l]


# ==================================================================#
# ==================================================================#
def send_mail(body="bcv002",
              attach=[],
              subject='Message from bcv002',
              to='rv.andres10@uniandes.edu.co'):
    import os
    content_type = {
        'jpg': 'image/jpeg',
        'gif': 'image/gif',
        'mp4': 'video/mp4'
    }
    if len(attach):  # Must be a list with the files
        enclosed = []
        for line in attach:
            format = line.split('.')[-1]
            enclosed.append('--content-type={} --attach {}'.format(
                content_type[format], line))
        enclosed = ' '.join(enclosed)
    else:
        enclosed = ''
    mail = 'echo "{}" | mail -s "{}" {} {}'.format(body, subject, enclosed, to)
    # print(mail)
    os.system(mail)


# ==================================================================#
# ==================================================================#
def single_source(tensor):
    import torch
    source = torch.ones_like(tensor)
    middle = 0  # int(math.ceil(tensor.size(0)/2.))-1
    source[middle] = tensor[0]
    return source


# ==================================================================#
# ==================================================================#
def slerp(val, low, high):
    """
  original: Animating Rotation with Quaternion Curves, Ken Shoemake
  https://arxiv.org/abs/1609.04468
  Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
  """
    import numpy as np
    omega = np.arccos(
        np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin(
        (1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


# ==================================================================#
# ==================================================================#
def target_debug_list(size, dim, config=None):
    import torch
    target_c = torch.zeros(size, dim)
    target_c_list = []
    for j in range(dim):
        target_c[:] = 0
        target_c[:, j] = 1
        target_c_list.append(to_var(target_c, volatile=True))
    return target_c_list


# ==================================================================#
# ==================================================================#
def TimeNow():
    import datetime
    import pytz
    return str(datetime.datetime.now(
        pytz.timezone('Europe/Amsterdam'))).split('.')[0]


# ==================================================================#
# ==================================================================#
def TimeNow_str():
    import re
    return re.sub(r'\D', '_', TimeNow())


# ==================================================================#
# ==================================================================#
def to_cpu(x):
    return x.cpu() if x.is_cuda else x


# ==================================================================#
# ==================================================================#
def to_cuda(x):
    import torch
    import torch.nn as nn
    if get_torch_version() > 0.3:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(x, nn.Module):
            x.to(device)
        else:
            return x.to(device)
    else:
        if torch.cuda.is_available():
            if isinstance(x, nn.Module):
                x.cuda()
            else:
                return x.cuda()
        else:
            return x


# ==================================================================#
# ==================================================================#
def to_data(x, cpu=False):
    if get_torch_version() > 0.3:
        x = x.data
    else:
        from torch.autograd import Variable
        if isinstance(x, Variable):
            x = x.data
    if cpu:
        x = to_cpu(x)
    return x


# ==================================================================#
# ==================================================================#
def to_parallel(main, input, list_gpu):
    import torch.nn as nn
    if len(list_gpu) > 1 and input.is_cuda:
        return nn.parallel.data_parallel(main, input, device_ids=list_gpu)
    else:
        return main(input)


# ==================================================================#
# ==================================================================#
def to_var(x, volatile=False, requires_grad=False, no_cuda=False):
    if not no_cuda:
        x = to_cuda(x)
    if get_torch_version() > 0.3:
        if requires_grad:
            return x.requires_grad_(True)
        else:
            return x

    else:
        from torch.autograd import Variable
        if isinstance(x, Variable):
            return x
        return Variable(x, volatile=volatile, requires_grad=requires_grad)
