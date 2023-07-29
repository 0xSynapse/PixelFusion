'''
Neural Style Transfer using TensorFlow's Pretrained Style Transfer Model 
https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2 

'''


import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import cv2
import os



model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")


# source: https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv 
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0): 
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

                     
def style_transfer(content_img,style_image, style_weight = 1, content_weight = 1, style_blur=False):
    content_img = unsharp_mask(content_img,amount=1)
    content_img = tf.image.resize(tf.convert_to_tensor(content_img,tf.float32)[tf.newaxis,...] / 255.,(512,512),preserve_aspect_ratio=True)
    style_img = tf.convert_to_tensor(style_image,tf.float32)[tf.newaxis,...] / 255.
    if style_blur:
        style_img=  tf.nn.avg_pool(style_img, [3,3], [1,1], "VALID")
    style_img = tf.image.adjust_contrast(style_img, style_weight)        
    content_img = tf.image.adjust_contrast(content_img,content_weight)     
    content_img = tf.image.adjust_saturation(content_img, 2)        
    content_img = tf.image.adjust_contrast(content_img,1.5)        
    stylized_img = model(content_img, style_img)[0]
    
    return Image.fromarray(np.uint8(stylized_img[0]*255))




title = "PixelFusion🧬"
description = "Gradio Demo for Artistic Neural Style Transfer. To use it, simply upload a content image and a style image. [Learn More](https://www.tensorflow.org/tutorials/generative/style_transfer)."
article = "</br><p style='text-align: center'><a href='https://github.com/0xsynapse' target='_blank'>GitHub</a></p> "


content_input = gr.inputs.Image(label="Upload an image to which you want the style to be applied.",)
style_input = gr.inputs.Image( label="Upload Style Image ",shape= (256,256), )
style_slider = gr.inputs.Slider(0,2,label="Adjust Style Density" ,default=1,)
content_slider = gr.inputs.Slider(1,5,label="Content Sharpness" ,default=1,)
# style_checkbox = gr.Checkbox(value=False,label="Tune Style(experimental)")


examples  = [
                ["Content/content_1.jpg","Styles/style_1.jpg",1.20,1.70,"style_checkbox"], 
                ["Content/content_2.jpg","Styles/style_2.jpg",0.91,2.54,"style_checkbox"],
                ["Content/content_3.png","Styles/style_3.jpg",1.02,2.47,"style_checkbox"]
            ]
interface = gr.Interface(fn=style_transfer,
                         inputs=[content_input,
                                style_input,
                                style_slider ,
                                content_slider,
                                # style_checkbox
                                ],
                         outputs=gr.outputs.Image(type="pil"),
                         title=title,
                         description=description,
                         article=article,
                         examples=examples,
                         enable_queue=True
                         )
    
    
interface.launch()