"""
Code used to save the keras model as an image for visualization
"""
from keras.layers import Convolution2D, Input
from keras.layers import Flatten, Dense
from keras.layers import Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pydot
except ImportError:
    # fall back on pydot if necessary
    import pydot

def model_vgg():
    """
    Using pre-trained VGG model without top layers.
    Reference https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    Model is trained with last four layer from VGG and with new three conv layers and 3 fully connected layers while freezing other layers.
    :return: model
    """
    in_layer = Input(shape=(160, 320, 3))
    model = VGG16(weights='imagenet', include_top=False, input_tensor=in_layer)
    for layer in model.layers[:15]:
        layer.trainable = False
    # Add last block to the VGG model with modified sub sampling.
    layer = model.outputs[0]
    # These layers are used for reducing the (5,10,512) sized layer into (1,5,512).
    layer = Convolution2D(512, 3, 3, subsample=(1, 1), activation='elu', border_mode='valid', name='block6_conv1')(
        layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 1), activation='elu', border_mode='same', name='block6_conv2')(
        layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 1), activation='elu', border_mode='valid', name='block6_conv3')(
        layer)
    layer = Flatten()(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(1024, activation='relu', name='fc1')(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(256, activation='relu', name='fc2')(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(1, activation='linear', name='predict')(layer)

    return Model(input=model.input, output=layer)


def model_to_dot(model, show_shapes=False, show_layer_names=True):
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if model.__class__.__name__ == 'Sequential':
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # first, populate the nodes of the graph
    for layer in layers:
        layer_id = str(id(layer))
        if show_layer_names:
            label = str(layer.name) + ' (' + layer.__class__.__name__ + ')'
        else:
            label = layer.__class__.__name__

        if show_shapes:
            # Build the label that will actually contain a table with the
            # input/output
            try:
                outputlabels = str(layer.output_shape)
            except:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)


    # second, add the edges
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                # add edges
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot(model, to_file='model.png', show_shapes=False, show_layer_names=True):
    dot = model_to_dot(model, show_shapes, show_layer_names)
    dot.write_png(to_file)

model = model_vgg()
plot(model, to_file='model.png',show_shapes=True)