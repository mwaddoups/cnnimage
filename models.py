from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.constraints import max_norm
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator

from utils.extra_metrics import f1score, f2score

class LabelModeller:
    def __init__(self, image_processor):
        self.labels_to_model = dict()
        self.image_processor = image_processor

    def fit_model_labels(self, labels, epochs=10, batch_size=256, gen_args=None, **kwargs):
        if not labels in self.labels_to_model:
            raise Exception('Missing model for label set {}.'.format(labels))       
 
        X_train, X_test, y_train, y_test = self.image_processor.get_train_test_data(labels)
        model = self.labels_to_model[labels]
        
        if gen_args is None:
            model.fit(X_train, y_train,
                      epochs = epochs,
                      batch_size = batch_size,
                      validation_data = (X_test, y_test),
                      **kwargs)
        else:
            train_gen = ImageDataGenerator(**gen_args)
            model.fit_generator(
                train_gen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch = int(len(X_train) / batch_size),
                epochs=epochs,
                validation_data = (X_test, y_test),
                **kwargs)
            

    def add_model_new(self, labels, output_activation):
        output_dim = len(labels)
        self.labels_to_model[labels] = self.base_model(output_dim, output_activation)
        return self.labels_to_model[labels]
    
    def add_model_from_file(self, labels, output_activation, weights_file, by_name):
        model = self.add_model_new(labels, output_activation)
        model.load_weights(weights_file, by_name=by_name)

    def save_model_weights(self, labels, output_file):
        self.labels_to_model[labels].save_weights(output_file)
 
    def base_model(self, output_dim, output_activation):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.image_processor.target_size,
                         name='conv1'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         name='conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         name='conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(32, activation='relu', 
                        kernel_constraint=max_norm(4.),
                        name='fc1'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu', 
                        kernel_constraint=max_norm(4.),
                        name='fc2'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation=output_activation))
        
        if output_activation == 'sigmoid':
            loss = binary_crossentropy
        elif output_activation == 'softmax':
            loss = categorical_crossentropy
        else:
            raise Exception('No default loss defined for {}'.format(output_activation))

        model.compile(loss=loss,
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model
