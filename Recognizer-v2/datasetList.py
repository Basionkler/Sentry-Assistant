import os

def dataset_list_update() :
	listOfFile = []
	for root, dirs, files in os.walk("./dataset", topdown=False):
	   for name in files:
	      listOfFile.append(os.path.join(root, name))
	print (listOfFile)
	f = open('datalist.txt','w')
	listOfFile.sort()
	for item in listOfFile:
	  f.write("%s\n" % item)

	f.close()
