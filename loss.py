from packages import *

def content_loss(content, target):
            return tf.reduce_mean(tf.square(content - target))

def gram_matrix(input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

def style_loss(style, gram):
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram))


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
        style_weight, content_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our layers.
        model_outputs = model(init_image)

        style_output_features = model_outputs[:num_style_layers]
        content_output_features = model_outputs[num_style_layers:]

        style_score = 0
        content_score = 0

        # style loss
        weight_per_style_layer = 1.0 / float(num_style_layers)
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * style_loss(comb_style[0], target_style)

        # content loss
        weight_per_content_layer = 1.0 / float(num_cont_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * content_loss(comb_content[0], target_content)

        style_score *= style_weight
        content_score *= content_weight

        # total loss
        loss = style_score + content_score
        return loss, style_score, content_score