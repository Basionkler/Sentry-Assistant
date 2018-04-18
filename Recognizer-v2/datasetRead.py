import face_recognition
import numpy as np
import os
import datasetList

def dataset_read() :
	listOfFile = []
	encodings = []
	facesName = []
	f = open('datalist.txt','r')
	i = 0
	for item in f.readlines():
		name = ""
		filenameExt = item.split("/")
		filename = filenameExt[len(filenameExt)-1].split(".")[0].split("_")
		del filename[len(filename)-1]
		
		name = " ".join(str(a) for a in filename)
		
		print(item)
		print(name)

		image = face_recognition.load_image_file(item.strip())
		face_encodings = face_recognition.face_encodings(image)

		if(len(face_encodings)>0):
			encoding = face_encodings[0]
			encodings.append(encoding)
			np.save('./trainedmodel/encoding%04d' % i,encoding)
			facesName.append(name)
			i+=1
		else:
			print(item+" can't be encode!")
	output = [encodings, facesName]

	f.close()

	f = open('./trainedmodel/knownNames.txt','w')
	for n in facesName:
		f.write("%s\n" % n)

	return output

def dataset_import_from_files() :
	encodings = []
	facesName = []
	i = 0
	f = open('./trainedmodel/knownNames.txt','r')
	facesList = f.readlines()
	for n in facesList:
		n = n[:-1]
		facesName.append(n)
	print(len(facesName))
	for root, dirs, files in os.walk("./dataset", topdown=False):
		for name in files:

			#image = face_recognition.load_image_file(item.strip())
			#face_encodings = face_recognition.face_encodings(image)
			
			print('processing ./trainedmodel/encoding%04d.npy' % i)
			encodings.append(np.load('./trainedmodel/encoding%04d.npy' % i))
			i+=1
	output = [encodings, facesName]

	f.close()

	return output

def dataset_update() :
	i=11
	encodings = []
	facesName = []
	f = open('./datalist.txt','r')
	facesList = f.readlines()
	datasetList.dataset_list_update()
	f = open('./datalist.txt','r')
	facesList2 = f.readlines()
	for face in facesList2:
		if face not in facesList:
			name = ""
			filenameExt = face.split("/")
			filename = filenameExt[len(filenameExt)-1].split(".")[0].split("_")
			del filename[len(filename)-1]
			
			name = " ".join(str(a) for a in filename)
			
			print(face)
			print(name)

			image = face_recognition.load_image_file(face.strip())
			face_encodings = face_recognition.face_encodings(image)

			if(len(face_encodings)>0):
				print('saving '+name)
				encoding = face_encodings[0]
				encodings.append(encoding)
				np.save('./trainedmodel/encoding%04d' % i,encoding)
				facesName.append(name)

				f = open('./trainedmodel/knownNames.txt','a')
				f.write("%s\n" % name)

				i+=1
			else:
				print(face+" can't be encode!")
		else:
			print(face+' is already a known face')
		
	output = [encodings, facesName]

	return output