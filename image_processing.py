from packages import *

def load(path, max_dim):
    img = Image.open(path)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS )
    img = im.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_img(path):
    img = load(path)
    return tf.keras.applications.vgg19.preprocess_input(img)
    
def deprocess_img(i):
    x = i.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    if len(x.shape) != 3:
        raise ValueError

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    return np.clip(x, 0, 255).astype('uint8')