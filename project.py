from neural2 import neuralNetwork
import skimage as ski
from skimage import data, io, filters, exposure
from skimage.filters import rank, threshold_minimum
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
from matplotlib import pylab as plt
import numpy as np
from numpy import array
import colorsys
from skimage import data
from matplotlib import pyplot as plt
import cv2
import sys
import math
from scipy import ndimage
from scipy import stats
from scipy.stats import hmean
import imutils
from sklearn.linear_model import LogisticRegression

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def resize(part):
        rows, cols = part.shape
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            part = cv2.resize(part, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            part = cv2.resize(part, (cols, rows))

        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        part = np.lib.pad(part,(rowsPadding,colsPadding),'constant')
        shiftx,shifty = getBestShift(part)
        shifted = shift(part,shiftx,shifty)
        gray = shifted
        return gray


def thresh(image, t):
    t = threshold_minimum(image)
    binary = (image > t) * 1.0
    return binary

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range (passnum):
            if alist[i][0]>alist[i+1][0]:
                temp = alist[i]
                alist[i]= alist[i+1]
                alist[i+1]=temp
    # return alist


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped

def findPaper(imgray, image):
    # plt.imshow(image)
    blurred = cv2.GaussianBlur(imgray,(3,3), 0)

    # plt.imshow(blurred, cmap="Greys_r")
    # plt.savefig('GaussianBlur.jpg')

    edged = cv2.Canny(blurred, 50, 200, 255)

    # plt.imshow(edged, cmap="Greys_r")
    # plt.savefig('Canny.jpg')

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

    # plt.imshow(edged, cmap="Greys_r")
    # plt.savefig('morphology.jpg')

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = []

    # cv2.drawContours(imgray, cnts, 0, (255,255,255),2)
    # plt.imshow(imgray)
    # plt.show()
    # plt.imshow(imgray, cmap="Greys_r")
    # plt.savefig('wykryte_kontury_paper.jpg')

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
        if len(approx) == 4:
            displayCnt = approx
            break
    # print(len(displayCnt))

    # cv2.drawContours(imgray, displayCnt, 0, (255,255,255),2)
    # plt.imshow(imgray)
    # plt.savefig('najwiekszy_kontur_papier.jpg')

    warped = four_point_transform(imgray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    # plt.imshow(output)
    # plt.savefig('wykryty_papier.jpg')



    return warped, output

def geo_mean(iterable):
    a = 1
    for i in iterable:
        a *= i[3]
    
    return a**(1.0/len(iterable))

def na_pinc(image_name):
    fig, ax = plt.subplots()
    im = cv2.imread('data_set/'+image_name+'.jpg')
    im = imutils.resize(im, height=500)
    # im = cv2.resize(im, (500, 500))
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # imgray = 255 - imgray
    plt.imshow(im)
    plt.savefig('original'+image_name+'.jpg')
    plt.show()

    dupa = imgray.copy()
    imgray, imCut = findPaper(dupa,im)

    imgray = 255 - imgray

    average = imgray.mean(axis=0).mean(axis=0)
    # imgray = adjust_gamma(imgray, 0.1)


    plt.imshow(imgray, cmap='Greys_r')
    plt.savefig('original'+image_name+'.jpg')
    plt.imshow(imCut, cmap='Greys_r')
    plt.savefig('cut'+image_name+'.jpg')

    x,y = imgray.shape
    print(x,y)
    print(0+int(x/10), x-int(x/10))
    imgray = imgray[0+int(x/10):x-int(x/10),0+int(y/30):y-int(y/30)]
    imCut = imCut[0+int(x/10):x-int(x/10),0+int(y/30):y-int(y/30),:]

    # plt.imshow(imCut)
    # plt.savefig('trimmed_paper.jpg')

    # plt.imshow(imgray, cmap='Greys_r')
    # plt.show()
    # plt.imshow(imCut, cmap='Greys_r')
    # plt.show()


    ## to dawalo sobie calkiem niezle rade ale niestety nie dla wszystkich
    # imgray = cv2.medianBlur(imgray,9, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel, iterations=2)

    ## usuwanie soli
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel, iterations=1)

    # plt.imshow(imgray, cmap='Greys_r')
    # plt.show()

    ret, thresh = cv2.threshold(imgray, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plt.imshow(thresh, cmap='Greys_r')
    # plt.savefig('thresh_paper')

    ## powiekszanie ksztaltow dla latwiejszego rozpoznania
    thresh = cv2.dilate(thresh, kernel, iterations = 1)

    # plt.imshow(thresh, cmap = 'Greys_r')
    # plt.savefig('dilated_thresh')

    imFinal = thresh.copy()

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_len = len(contours)
    print('len of contours = ' + str(cont_len))


    # cv2.drawContours(imCut, contours, -1, (255,0,0),2)
    # plt.imshow(imCut)
    # plt.savefig('kontury_znakow.jpg')
    # plt.show()


    meanOfContours = 0
    for cnt in contours:
        meanOfContours += cv2.contourArea(cnt)
    meanOfContours = meanOfContours/len(contours)

    # bubbleSort(contours)

    secondaryTable = []
    meanH = 0
    i = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > meanOfContours/4:
            x,y,w,h = cv2.boundingRect(cnt)
            meanH += h
            i += 1
            secondaryTable.append((x,y,w,h))
    meanH = meanH/i
    print('meanH = ' + str(meanH))
    meanH2 = geo_mean(secondaryTable)
    print('meanH2 = ' + str(meanH2))
    meanH3 = hmean([h for x,y,w,h in secondaryTable if h>0])
    print('meanH3 = ' + str(meanH3))
    bubbleSort(secondaryTable)


    numbers = []
    operators = []
    artifacts = []
    numberSquares = []
    operatorSquares = []
    j = 0
    k = 0

    print("DUPA")
    heights = [h for x,y,w,h in secondaryTable]
    print(heights)
    maxH = np.max(np.asarray(heights))
    k = 100/maxH
    heights = [h*k for h in heights]
    print(heights)
    print(meanH2*k)

    for i in range(len(secondaryTable)):
        x,y,w,h = secondaryTable[i]
        h = heights[i]
        print(x ,y ,w, h)

        part = imFinal[int(y):int(y+h),int(x):int(x+w)].copy()

        # cv2.rectangle(imCut,(x,y),(x+w,y+h),(255,0,0),2)

        res = resize(part)
        res = (res/255.0 *0.99)+0.01

        if(h > 50):
            numbers.append(res)
            numberSquares.append(secondaryTable[i])

        if( h >=13  and h <= 50):
            operators.append(res)
            operatorSquares.append(secondaryTable[i])

        if(h<13):
            artifacts.append(res)


        # if(h > meanH2/2):
        #     numbers.append(res)
        #     numberSquares.append(secondaryTable[i])
        #     # plt.imshow(res, cmap='Greys_r')
        #     # plt.savefig('number'+str(j)+'.jpg')
        #     j += 1
        # if(h<meanH2/4):
        #     artifacts.append(res)
        # """h >= meanH2/4 and """
        # if(h < meanH2/2):
        #     operators.append(res)
        #     operatorSquares.append(secondaryTable[i])
        #     plt.imshow(res, cmap='Greys_r')
        #     plt.savefig('operator'+str(k)+'.jpg')
        #     k += 1


    ## TODO: wymyslic lepszy sposob dzielenia na operatory i numery
    print('numbers len ' + str(len(numbers)))
    print('operators len ' + str(len(operators)))

    ## bylo
    # plt.imshow(imFinal, cmap='Greys_r')
    # plt.savefig(image_name+'.pdf')
    # plt.show()

    # plt.imshow(imCut)
    # plt.show()

    # plt.subplot(2,1,1)
    # plt.imshow(im)
    # plt.imshow(imgray, cmap='Greys_r')
    
    # plt.subplot(2,1,1)
    # plt.imshow(thresh, cmap='Greys_r')
    # plt.subplot(2,1,2)
    # plt.imshow(imFinal, cmap='Greys_r')
    # plt.show()
    return numbers, operators, numberSquares, operatorSquares, imCut

# def process(img):


def load(path):
    data_file = open(path, "r")
    data_list = data_file.readlines()
    data_file.close()
    return data_list

if __name__ == "__main__":
    ## to tylko do generowania wszystkich pdfow naraz. zakomentowac jak cos.
    # names = ['10','11a','11b','11c','12','13','14','15','16','17','18','19']
    # names = ['21a','21b','21c','21d']
    names = []
    names.append(sys.argv[1])
    print(names[0])
    for name in names:

        # numbers, numberSquares, imCut = na_pinc(sys.argv[1])
        numbers, operators, numberSquares, operatorSquares, imCut = na_pinc(name)
        print('imCut shape = '+str(imCut.shape))        
        imCut = cv2.copyMakeBorder(imCut,60,60,0,0,cv2.BORDER_CONSTANT, value=(255,255,255))
        print('imCut shape = '+str(imCut.shape))
        for i in range (len(numberSquares)):
            tempX = numberSquares[i][0]
            tempX += 0
            tempY = numberSquares[i][1]
            tempY += 60
            tempW = numberSquares[i][2]
            tempH = numberSquares[i][3]
            numberSquares[i]= (tempX,tempY,tempW,tempH)

        for i in range (len(operatorSquares)):
            tempX = operatorSquares[i][0]
            tempX += 0
            tempY = operatorSquares[i][1]
            tempY += 60
            tempW = operatorSquares[i][2]
            tempH = operatorSquares[i][3]
            operatorSquares[i]= (tempX,tempY,tempW,tempH)

        numbers = np.asarray(numbers)
        operators = np.asarray(operators)

        input_nodes = 784
        hidden_nodes = 150
        hidden2_nodes = 150
        output_nodes = 10
        learning_rate = 0.2
        epochs = 1

        n = neuralNetwork(input_nodes,hidden_nodes,hidden2_nodes,output_nodes,learning_rate,epochs)
        sym = neuralNetwork(input_nodes,100,100,4,learning_rate,epochs)

        n.deserialize("weights.pkl")
        sym.deserialize("symbols.pkl")

        test_numbers = []
        test_operators = []
        for item in numbers:

            pom = item.reshape((784,))
            # pom = np.flip(pom,0)

            # plt.imshow(item, cmap='Greys_r')
            # plt.show()
            test_numbers.append(pom)
            # print(item.shape)

        results = []
        results2 = []
        for item in test_numbers:
            output = np.argmax(n.query(item))
            results.append(output)

        for i in range(len(results)):
            imX, imY, imZ = imCut.shape
            x,y,w,h = numberSquares[i]
            cv2.rectangle(imCut, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imCut, str(results[i]), (x, y),
            cv2.FONT_HERSHEY_PLAIN, (imX-120)/30, (0, 255, 0), 2)


        for item in operators:
            pom = item.reshape((784,))
            test_operators.append(pom)      

        for item in test_operators:
            output = np.argmax(sym.query(item))
            # output = sym.query(item)            
            results2.append(output)            

        for i in range(len(results2)):
            imX, imY, imZ = imCut.shape
            x,y,w,h = operatorSquares[i]
            cv2.rectangle(imCut, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imCut, str(results2[i]), (x, y),
            cv2.FONT_HERSHEY_PLAIN, (imX-120)/30, (255, 0, 0), 2)

        print('numbers: ' + str(results))
        print('operators: ' + str(results2))
        plt.imshow(imCut)
        plt.savefig('output'+name+'.jpg')
        # plt.savefig('klasified_shapes_'+name+'.jpg')
        plt.show()

        #+ 0
        #- 1
        #/ 2
        #/ 3