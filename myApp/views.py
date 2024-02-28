import base64

from django.core.files.images import get_image_dimensions
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse
from django.shortcuts import render, redirect
import numpy as np
from PIL import Image
import io
from model.yolo_detection_images import YoloModel
import cv2
import shutil
import sys
from .models import *
# sys.path.append("../..")

y = YoloModel()
csvData = []
ress = None
def home(request):
    return render(request, 'index.html')

def switchAuto(request):
    return render(request,'home.html')

def switchMan(request):
    return render(request, 'manual_annotate.html')

# function to upload images
def uploadImages(request):

    if request.headers.get('x-requested-with') == 'XMLHttpRequest' and request.method =='POST':
        # function to clear result data
        clearResults()
        images = request.FILES.getlist('images')
        if len(images) > 0:
            if len(y.testImgs) > 0:
                y.testImgs.clear()
                y.testImgNames.clear()
            for img in images:
                # converting input image to cv format
                npimg = np.fromstring(img.file.read(), np.uint8)
                imgRGB = cv2.cvtColor(cv2.imdecode(npimg, cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                y.testImgs.append(imgRGB)
                y.testImgNames.append(img.name)
            return JsonResponse({'msg':'Uploaded successfully'},safe=False)
        else:
            return JsonResponse({'msg':'No Image uploaded'},safe=False)
    return render(request,'home.html')


# function to select the model
def loadModel(request):
    index = request.POST.get('index', None)
    if len(index) > 0:
        print(index)
        if index == '1':
            y.init_values('coco')
        elif index == '2':
            y.init_values('chess')
        elif index == '3':
            y.init_values('TechM')


    return render(request,'home.html')

# function to add custom model files in the project
def customModel(request):
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' and request.method == 'POST':
        files = request.FILES.getlist('model_files')
        if len(files) > 0:
            clearCustomModelFiles()
            for file in files:
                CustomModelFiles(files=file).save()
            # initializing custom model after files are saved
            y.init_values('custom')
            return JsonResponse({'msg': 'Uploaded successfully'}, safe=False)
        else:
            return JsonResponse({'msg': 'No Files uploaded'}, safe=False)
    return render(request,'home.html')

# function to run model
def runmodel(request):
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        if y.testImgs:
            for ind, img in enumerate(y.testImgs):
                origImg = Image.fromarray(img.astype("uint8"))
                # res[0] = image, res[1] = labels, res[2] = bounding boxes, res[3] = image name, res[4] = image width and height, res[5] = dict of images and bounding boxes
                res = y.runModel(img, ind)
                res[0] = Image.fromarray(res[0].astype("uint8"))
                resImgBytes = io.BytesIO()
                origImgBytes = io.BytesIO()
                origImg.save(origImgBytes, "JPEG")
                res[0].save(resImgBytes, "JPEG")
                csvData.append([res[3], res[2], res[1], resImgBytes.getvalue(), res[4],res[5]])
                storeAndLabelImg(res[1], origImgBytes,res[3],res[5])
            return JsonResponse({'msg':'Success'},safe=False)
        else:
            return JsonResponse({'msg': 'No image uploaded. Please upload again!'}, safe=False)
    return render(request,'home.html')


# function to display images to user
def showResults(request):
    if request.method == 'POST':
        res = []
        for data in csvData:
            # converting imgs to base64 format
            res.append([str(base64.b64encode(data[3])),data[2]])
        return JsonResponse({'images':res},safe=False)
        # return render(request, 'home.html', {'imagess': data})
    return render(request,'home.html')


# function to convert uploaded images to base64 format in manual annotation
def manualUpload(request):

    if request.headers.get('x-requested-with') == 'XMLHttpRequest' and request.method == 'POST':
        imgs = request.FILES.getlist('images')
        sentImgs = []
        width = 0
        height = 0
        dimens = []
        for img in imgs:
            width,height = get_image_dimensions(img)
            dimens.append([width,height])
            img_base64 = base64.b64encode(img.read())
            sentImgs.append([img.name,str(img_base64)])
        return JsonResponse({'imgs':sentImgs,'dimen':dimens},safe=False)


    return render(request, 'manual_annotate.html')


# function to download result data
def download(request):
    import zipfile
    import csv
    from django.http import HttpResponse
    response = None

    if request.method == 'POST':
        value = request.POST['download_options']
        if value == 'Zip':                      # download as zip format
            filePaths = []
            dir_name = 'media/result'
            for root, directories, files in os.walk(dir_name):
                for filename in files:
                    # Create the full filepath by using os module.
                    filePath = os.path.join(root, filename)
                    filePaths.append(filePath)

            response = HttpResponse(content_type='application/zip')
            zf = zipfile.ZipFile(response, 'w')
            for file in filePaths:
                zf.write(file)
            response['Content-Disposition'] = f'attachment; filename=result.zip'

        elif value == 'darknet':                 # download as darknet format
            filePaths = []
            dir_name = 'media/all_training_images'
            for root, directories, files in os.walk(dir_name):
                for filename in files:
                    # Create the full filepath by using os module.
                    filePath = os.path.join(root, filename)
                    filePaths.append(filePath)

            response = HttpResponse(content_type='application/zip')
            zf = zipfile.ZipFile(response, 'w')
            for file in filePaths:
                zf.write(file)
            response['Content-Disposition'] = f'attachment; filename=result.zip'

        elif value == 'CSV':  # download as csv
            response = HttpResponse(content_type='text/csv')
            writer = csv.writer(response)
            # columns in csv file
            writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

            # 0-name, 1-bounding box, 2-labels, 4-width and height
            for data in csvData:

                imgName = data[0]  # image name
                boundingBoxes = data[1]
                size = data[4]
                for label in boundingBoxes:
                    for box in boundingBoxes[label]:
                        writer.writerow([imgName,size[0],size[1],label,box[0],box[1],box[2],box[3]])

            response['Content-Disposition'] = f'attachment; filename=result.csv'

    if response:
        return response
    return render(request,'home.html')


# function to clear result data
def clearResults():
    csvData.clear()
    if os.path.isdir('media/result'):
        Result.objects.all().delete()
        TrainingData.objects.all().delete()
        shutil.rmtree('media/result')
        shutil.rmtree('media/all_training_images')

# function to clear custom model files
def clearCustomModelFiles():
    if len(CustomModelFiles.objects.all()) > 0:
        CustomModelFiles.objects.all().delete()
        shutil.rmtree('model/custom')

# function to label and store images
def storeAndLabelImg(labels,buff,origImageName,allBoundingBoxes):
    if not os.path.isdir('media/result'):
        os.makedirs('media/result')

    storeTrainingData(buff,origImageName,allBoundingBoxes)


    for i in range (0,len(labels)):

        labelName = labels[i]
        if labelName:
            # create path for output img as per label i.e labelName variable
            path = 'media/result' + '/' + labelName

            if not os.path.isdir(path):
                os.mkdir(path)

            imageName = origImageName

            # converting result image to InMemoryUploadedFile type inorder to save info of it in db
            img = InMemoryUploadedFile(buff, None,
                                           imageName,
                                           'image/jpeg',
                                           len(str(buff)), None)

            Result(imageLabel=labelName, imageFile=img).save()

            # write txt file as per folder structure
            s = "%s %s %s %s %s \n"  # the first number is the identifier of the class. If you are doing multi-class, make sure to change that
            txtFileName = os.path.splitext(origImageName)[0]
            file_name = "{}/{}.txt".format(path, txtFileName)   # specify the name of the folder and get a file name
            with open(file_name, 'a') as file:  # append lines to file
                for i in allBoundingBoxes[labelName]:
                    new_line = (s % tuple(i))
                    file.write(new_line)

        else:
            break



def storeTrainingData(buff,origImageName,boundingBox):
    if not os.path.isdir('media/all_training_images'):
        os.mkdir('media/all_training_images')
    # print(boundingBox)
    path = 'media/all_training_images'
    txtFileName = os.path.splitext(origImageName)[0]
    file_name = "{}/{}.txt".format(path, txtFileName)  # specify the name of the folder and get a file name

    s = "%s %s %s %s %s \n"  # the first number is the identifier of the class. If you are doing multi-class, make sure to change that
    with open(file_name, 'a') as file:  # append lines to file
        for label in boundingBox:
            for box in boundingBox[label]:
                print(box)
                new_line = (s % tuple(box))
                file.write(new_line)
        # for i in boundingBox:
        #     new_line = (s % tuple(i))
        #     file.write(new_line)

    # save image file
    img = InMemoryUploadedFile(buff, None,
                               origImageName,
                               'image/jpeg',
                               len(str(buff)), None)
    TrainingData(imageFile=img).save()

    # save label names file
    shutil.copy(y.currLabelsPath,'media/all_training_images')