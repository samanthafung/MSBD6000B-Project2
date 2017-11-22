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

# later...
img_width, img_height = 64, 64
test_data_dir = './data/test'
batch_size = 32

# load json and create model
json_file = open('smart_pig/smart_pig_model_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("smart_pig/sp_try3.h5")
print("Loaded model from disk")
print(loaded_model.summary())

# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=False,
#     save_to_dir='./data/ans/',
#     save_format = 'jpeg')

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# predict_ans = loaded_model.predict_generator(test_generator, 551, workers=12)

# scores = loaded_model.evaluate_generator(test_generator, 551//batch_size, workers=12)
outF = open("project2_10001.txt", "w")
with open('test.txt','rb') as f:
    # i = 0
    for line in f :
        img = load_img(line.split()[0])
        x = img_to_array(img.resize((img_width, img_height)))
        x = x.reshape((1,) + x.shape)
        temp_ans = loaded_model.predict(x, batch_size=batch_size)
        print(line.split()[0].decode('UTF-8') + ' ' + str(temp_ans[0].argmax()))
        outF.write(line.split()[0].decode('UTF-8') + ' ' + str(temp_ans[0].argmax()) + "\n")
        # val_x = img_to_array(img.resize((img_width, img_height))
        # x = x.reshape((1,) + x.shape)
        # y.append ( line.split()[1] )
        # print(line.rsplit()[1].decode('UTF-8'))
        # img.save(os.path.join('./data2/validation/' + str(line.split()[1].decode('UTF-8')) +'/__00'+ str(i) +'.jpg'), 'JPEG')
        # i += 1
outF.close()

# img = load_img('./data/test/16018886851_c32746cb72.jpg')
# x = img_to_array(img.resize((img_width, img_height)))
# x = x.reshape((1,) + x.shape)

# temp_ans = loaded_model.predict(x, batch_size=batch_size)
# print(loaded_model.get_config() )
# print(temp_ans)
# print(type(temp_ans[0]))
# print(max(temp_ans[0]))
# print(temp_ans[0].argmax())
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))
