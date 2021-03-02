from loss import *
from image_processing import *
import sys


def main(cp, sp, max_dim, num_iterations=10):    
    def get_model():
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_out = [vgg.get_layer(layer_name).output for layer_name in style_layers]
        content_out = [vgg.get_layer(layer_name).output for layer_name in content_layers]
        model_out = style_out + content_out # all in one to easy access using num_layers

        return models.Model(vgg.input, model_out)

    

    def feature(model, content_path, style_path):
        content = load(content_path, max_dim)
        style = load(style_path, max_dim)

        style_out = model(style)
        content_out = model(content)
        

        c_features = [layer[0] for layer in content_out[num_style_layers:]]
        s_features = [layer[0] for layer in style_out[:num_style_layers]]

        return s_features, c_features
   
    # Compute gradients according to input image
    def compute_grads(cfg):
        with tf.GradientTape() as tape:
            all_loss = compute_loss(**cfg)
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss

    def run_style_transfer(content_path,
                            style_path,
                            num_iterations=1000,
                            content_weight=1e3,
                            style_weight=1e-2):
        # We don't train any layers of our model
        model = get_model()
        for layer in model.layers:
            layer.trainable = False
        
        style_features, content_features = feature(model, content_path, style_path)
        gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

        # Set image
        init_image = load(content_path, max_dim)
        init_image = tf.Variable(init_image, dtype=tf.float32)
        # Adam Optimizer
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        best_loss, best_img = float('inf'), None

        # Create config
        loss_weights = (style_weight, content_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        # main loop
        count = num_iterations//99
        for i in range(num_iterations):
            percent = i//count
            print(f'\r[{"#"*(percent//2)}{" "*(50-percent//2)}]{percent}%', end='')
            grads, all_loss = compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)

            if loss < best_loss:
                best_loss = loss
                best_img = deprocess_img(init_image.numpy())

        return best_img, best_loss

    best, best_loss = run_style_transfer(cp, sp)
    imr = Image.fromarray(best)
    return imr



if __name__ == '__main__':
    root = Tk()
    root.withdraw()
    try:
        resolution = int(input('Enter resolution of output image(256> <2048): '))
        if 2048<resolution<256:
            raise ValueError
        content = askopenfilename(title='Select content image', filetypes=(("jpeg files","*.jpg"),))
        style = askopenfilename(title='Select style image', filetypes=(("jpeg files","*.jpg"),))
        final = main(content, style, resolution)
        print("\nDone")
        
        result = input("Enter the name of output file: ")
        final.save(f"{result}.jpg")
    except ValueError:
        print('Resolution should be number between 256 and 2048')
    except:
        print('Something went wrong...\nPlease check youre inputs and run program again')
    # can set 4th parameter num of iterations if needed