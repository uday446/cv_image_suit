import sys
from cv_image_suit.Logging_Layer.logger import App_Logger
import numpy
from flask import Flask, render_template, request, jsonify, send_file
from cv_image_suit.com_in_ineuron_ai_utils.utils import decodeImage
from flask_cors import CORS
from cv_image_suit.Prediction_Layer.predict import tfpredict
import json
import webbrowser
from threading import Timer
from cv_image_suit.Training_Layer.train_engine import tftrainer
from cv_image_suit.Experiment_Layer.model_finder import modelfinder
from os import listdir
import os
from tensorboard import program
from cv_image_suit.com_in_ineuron_ai_utils.utils import batch_validate
from cv_image_suit.Exception_Layer.exception import GenericException

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = tfpredict(self.filename)
        self.tftraining = tftrainer()
        self.modelfinder = modelfinder("Config_Layer/experiment_inputs_configs.json")
        self.logger = App_Logger()



app = Flask(__name__) # initializing a flask app
CORS(app)

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/input2',methods=['GET'])  # route to display the home page
def input_form2():
    return render_template("input_experiments.html")

@app.route('/input3',methods=['GET'])  # route to display the home page
def input_form3():
    exps=[]
    if os.path.isdir(os.getcwd() + "/Tensorboard/logs/"):
        for x in listdir(os.getcwd() + "/Tensorboard/logs/"):
            exps.append(x)
    if not isinstance(exps, list):
        exps = [exps]
    return render_template("experiment_log.html",exps=exps)

@app.route('/input',methods=['GET'])  # route to display the home page
def input_form():
    model_list = []
    if os.path.isdir(os.getcwd()+"/New_trained_model"):
        for x in listdir(os.getcwd()+"/New_trained_model"):
            model_list.append(x)
    return render_template("input_form.html",model_list=model_list)

@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = os.getcwd()+"/Config_Layer/experiment_result.json"
    return send_file(path, as_attachment=True)

@app.route('/experiment',methods=['POST','GET'])
def experiment_func():
    if request.method == 'POST':
        try:
            clApp = ClientApp()
            # Data config
            TRAIN_DATA_DIR = request.form['TRAIN_DATA_DIR']
            VALID_DATA_DIR = request.form['VALID_DATA_DIR']
            CLASSES = int(request.form['CLASSES'])
            IMAGE_SIZE = request.form['IMAGE_SIZE']
            AUGMENTATION = request.form['AUGMENTATION']
            BATCH_SIZE = request.form['BATCH_SIZE']
            PERCENT_DATA = int(request.form['PERCENTAGE_DATA'])

            BATCH = BATCH_SIZE.split(',')
            BATCH = [int(j) for j in BATCH]
            BATCH_SIZE = numpy.sort(BATCH)
            BATCH_SIZE = [int(j) for j in BATCH_SIZE]
            size = IMAGE_SIZE.split(',')
            size = [int(j) for j in size]
            if PERCENT_DATA > 100:
                return render_template('input_experiments.html',error=["PERCENT_DATA cannot be more than 100"])
            size = numpy.sort(size)
            size = [int(j) for j in size]
            allowed_batch = batch_validate(TRAIN_DATA_DIR,VALID_DATA_DIR,PERCENT_DATA)
            if BATCH_SIZE[0] > allowed_batch:
                return render_template('input_experiments.html', error=["Please provide smaller batch size"])
            elif size[1] < 71 and size[0] != 3:
                return render_template('input_experiments.html', error=["Image size must be larger than 71x71 and last value must 3"])

            # Model config
            #MODEL_OBJ = request.form['MODEL_OBJ']
            MODEL_OBJ = request.form.getlist('MODEL_OBJ')
            EXP_NAME = request.form['EXP_NAME']
            FREEZE_ALL = request.form['FREEZE_ALL']
            FREEZE_TILL = int(request.form['FREEZE_TILL'])
            OPTIMIZER = request.form.getlist('OPTIMIZER')
            EPOCHS = int(request.form['EPOCHS'])

            configs = {
                "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
                "VALID_DATA_DIR": VALID_DATA_DIR,
                "AUGMENTATION": AUGMENTATION,
                "CLASSES": CLASSES,
                "IMAGE_SIZE": IMAGE_SIZE,
                "BATCH_SIZE": BATCH_SIZE,
                "MODEL_OBJ": MODEL_OBJ,
                "EXP_NAME": EXP_NAME,
                "EPOCHS": EPOCHS,
                "FREEZE_ALL": FREEZE_ALL,
                "FREEZE_TILL": FREEZE_TILL,
                "OPTIMIZER": OPTIMIZER,
                "PERCENT_DATA": PERCENT_DATA
            }

            with open("Config_Layer/experiment_inputs_configs.json", "w") as f:
                json.dump(configs, f)
                f.close()

            Phase = "MODEL"
            best_model = clApp.modelfinder.find_model(phase=Phase)

            #Phase = "IMAGE_SIZE"
            #best_model = clApp.modelfinder.find_model(phase=Phase)

            #Phase = "BATCH_SIZE"
            #best_model = clApp.modelfinder.find_model(phase=Phase)

            #Phase = "OPTIMIZER"
            #best_model = clApp.modelfinder.find_model(phase=Phase)

            exps = []
            if os.path.isdir(os.getcwd() + "/Tensorboard/logs/"):
                for x in listdir(os.getcwd() + "/Tensorboard/logs/"):
                    exps.append(x)
            if not isinstance(exps,list):
                exps = [exps]

            return render_template('experiment_output.html', model_list=best_model,exps=exps)
        except Exception as e:
            exception = GenericException(
                "Failed during training in module [{0}] class [{1}] method[{2}]"
                    .format(ClientApp.__module__, ClientApp.__name__,
                            train_func.__name__))
            file = open("Logs/logs.txt", 'a+')
            exception_msg = exception.error_message_detail(str(e), sys)
            clApp.logger.log(file, "Exception occured %s " % exception_msg)
            file.close()
            raise Exception(exception_msg)
            #return 'something is wrong'+str(e)

    else:
        return render_template('index.html')

@app.route('/train',methods=['POST','GET']) # route to show the predictions in a web UI
def train_func():
    if request.method == 'POST':
        try:
            clApp = ClientApp()
            # Data config
            TRAIN_DATA_DIR = request.form['TRAIN_DATA_DIR']
            VALID_DATA_DIR = request.form['VALID_DATA_DIR']
            CLASSES =int(request.form['CLASSES'])
            IMAGE_SIZE = request.form['IMAGE_SIZE']
            AUGMENTATION = request.form['AUGMENTATION']
            BATCH_SIZE = int(request.form['BATCH_SIZE'])


            # Model config
            MODEL_OBJ = request.form['MODEL_OBJ']
            MODEL_NAME = request.form['MODEL_NAME']
            FREEZE_ALL = request.form['FREEZE_ALL']
            FREEZE_TILL = request.form['FREEZE_TILL']
            OPTIMIZER = request.form['OPTIMIZER']
            EPOCHS = int(request.form['EPOCHS'])
            RESUME = request.form['RESUME']

            size = IMAGE_SIZE.split(',')
            size = [int(j) for j in size]
            size = numpy.sort(size)
            size = [int(j) for j in size]
            PERCENT_DATA=100
            allowed_batch = batch_validate(TRAIN_DATA_DIR,VALID_DATA_DIR,PERCENT_DATA)
            if BATCH_SIZE > allowed_batch:
                return render_template('input_experiments.html', error=["Please provide smaller batch size"])
            elif size[1] < 71 and size[0] != 3:
                return render_template('input_experiments.html', error=["Image size must be larger than 71x71 and last value must 3"])

            configs = {
                "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
                "VALID_DATA_DIR": VALID_DATA_DIR,
                "AUGMENTATION": AUGMENTATION,
                "CLASSES": CLASSES,
                "IMAGE_SIZE": IMAGE_SIZE,
                "BATCH_SIZE": BATCH_SIZE,
                "MODEL_OBJ": MODEL_OBJ,
                "MODEL_NAME": MODEL_NAME,
                "EPOCHS": EPOCHS,
                "FREEZE_ALL": FREEZE_ALL,
                "FREEZE_TILL": FREEZE_TILL,
                "OPTIMIZER": OPTIMIZER,
                "RESUME": RESUME
            }

            with open("Config_Layer/configs.json", "w") as f:
                json.dump(configs, f)
                f.close()

            hist = clApp.tftraining.train()
            my_list = ["Traning Completed  ",hist]
            return render_template('input_form.html',output = my_list)

        except Exception as e:
            exception = GenericException(
                "Failed during training in module [{0}] class [{1}] method[{2}]"
                .format(ClientApp.__module__, ClientApp.__name__,
                         train_func.__name__))
            file = open("Logs/logs.txt", 'a+')
            exception_msg = exception.error_message_detail(str(e), sys)
            clApp.logger.log(file, "Exception occured %s " % exception_msg)
            file.close()
            raise Exception(exception_msg)
            #print('The Exception message is: ',e)
            #return 'something is wrong'

    else:
        return render_template('index.html')


@app.route('/mid',methods=['GET','POST'])  # route to display the home page
def pred():
    model_list = []
    if os.path.isdir(os.getcwd() + "/New_trained_model"):
        for x in listdir(os.getcwd()+"/New_trained_model"):
            model_list.append(x)
    return render_template("mid_form.html",model_list=model_list,error="")

@app.route('/test',methods=['GET','POST'])  # route to display the home page
def predcit():
    model_list = []
    if os.path.isdir(os.getcwd() + "/New_trained_model"):
        for x in listdir(os.getcwd() + "/New_trained_model"):
            model_list.append(x)
    MODEL_NAME = request.form['MODEL_NAME']
    model_found=0
    for x in model_list:
        if MODEL_NAME == x:
            model_found = 1
        else:
            model_found = 0
    if model_found == 0:
        return render_template("mid_form.html",model_list=model_list,error=["Please Provide Model From Above List"])
    configs = {
        "MODEL_NAME": MODEL_NAME
    }

    with open("Config_Layer/pred_configs.json", "w") as f:
        json.dump(configs, f)
        f.close()
    return render_template("predict.html")

@app.route('/logs',methods=['GET','POST'])  # route to display the home page
def log():
    tracking_address = "Tensorboard/logs/fit"
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    webbrowser.open_new(url)
    return render_template("input_form.html")

@app.route('/explogs',methods=['GET','POST'])  # route to display the home page
def log2():
    if request.method == 'POST':
        try:
            dir = request.form['exp']
            tracking_address = "Tensorboard/logs/"+dir
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', tracking_address])
            url = tb.launch()
            webbrowser.open_new(url)
            print(f"Tensorflow listening on {url}")
            return render_template("experiment_log.html",url = url)
        except Exception as e:
            print(str(e))
            return render_template("experiment_log.html")

@app.route("/predict", methods=['POST'])
def predictRoute():
    clApp = ClientApp()
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predictiontf()
    return jsonify(result)



def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')


def start_app():
    Timer(3, open_browser).start()
    app.run(host="127.0.0.1", port=8080)



if __name__ == "__main__":
    start_app()
