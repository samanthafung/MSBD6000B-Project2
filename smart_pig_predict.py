from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import numpy
import os


# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

img_width, img_height = 64, 64
test_data_dir = './data/test'
batch_size = 32

# load json and create model
json_file = open('smart_pig/smart_pig_model_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("smart_pig/sp_try4.h5")
print("Loaded model from disk")
print(loaded_model.summary())


# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

outF = open("project2_20057566.txt", "w")
with open('test.txt','rb') as f:

    for line in f :
        img = load_img(line.split()[0])
        x = img_to_array(img.resize((img_width, img_height)))
        x = x.reshape((1,) + x.shape)
        temp_ans = loaded_model.predict(x, batch_size=batch_size)
        print(line.split()[0].decode('UTF-8') + ' ' + str(temp_ans[0].argmax()))
        outF.write(line.split()[0].decode('UTF-8') + ' ' + str(temp_ans[0].argmax()) + "\n")

outF.close()


