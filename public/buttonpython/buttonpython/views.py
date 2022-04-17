from msilib.schema import File
from django.shortcuts import render
from subprocess import run,PIPE
from django.views.decorators.csrf import csrf_exempt
import requests
from django.core.files.storage import FileSystemStorage

import sys
def button(request):
    return render(request,'homepage.hbs')

@csrf_exempt 
def submit(request):
    img = request.FILES['myfile']
    fs=FileSystemStorage()
    filename=fs.save(img.name,img)
    fileurl=fs.open(filename)
    templateurl=fs.url(filename)
    print("file raw url:",filename)
    print("file full url:",fileurl)
    print("template url:",templateurl)
    out = run([sys.executable,'.//views//python test.py',str(fileurl),str(filename)],stdout=PIPE)
    out = str(out.stdout.decode('utf-8'))
    info = out.split(" ")
    img = info[0]
    top = info[1]
    bottom = info[2]
    print("after",img,top,bottom)
    return render(request,'homepage.hbs',{'top':top,'bottom':bottom,'img':img})