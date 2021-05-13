from numpy import ndarray, array, asarray, append, round, hstack, mean
from matplotlib.pyplot import hist, title, show, plot
from cv2 import imread, imshow, waitKey, destroyAllWindows,calcHist, resize, convertScaleAbs, equalizeHist, createCLAHE, \
    cvtColor,  COLOR_BGR2GRAY, THRESH_BINARY, threshold
from typing import List, Tuple, Callable
class ImageEditor:

    def editImagArray(self, imgArray: ndarray, equa_method: str, scale: int = 30):
        bwimage: ndarray = self.__resizeImage(self.__convertImgToBW(imgArray=imgArray), scale_percent=scale)
        if(equa_method == 'clahe'):
            return self.__claheEqualization(bwimage);
        elif(equa_method=='equalizationHist'):
            return self.__equalizeHistagram(bwimage);
        elif(equa_method=='binary'):
            return self.__convertToBinary(bwimage);
        else:
            return bwimage

    def __convertImgToBW(self, imgArray: ndarray) -> ndarray:
        grayImgArray: ndarray = self.__convertImgToGS(imgArray)
        return threshold(grayImgArray, 110, 255, THRESH_BINARY)[1];
    @staticmethod
    def __convertImgToGS(imgArray: ndarray)->ndarray:
        return cvtColor(imgArray, COLOR_BGR2GRAY);
    @staticmethod
    def __resizeImage(imgArray: ndarray, scale_percent: int ) -> ndarray:

        width: int = int(imgArray.shape[1]* scale_percent /100);
        height: int = int(imgArray.shape[0]*scale_percent/100);
        rescale_v: Tuple[int, int] = (width, height)
        return resize(imgArray, rescale_v)
        #return imgArray.reshape([rescale_v[0], rescale_v[1]])

    @staticmethod
    def __claheEqualization(imgArray: ndarray) -> ndarray:
        clahe: createCLAHE = createCLAHE(clipLimit=2.0, tileGridSize=(8,8));
        return clahe.apply(imgArray);
    @staticmethod
    def __equalizeHistagram(imgArray: ndarray) -> ndarray:
        equali: ndarray = equalizeHist(imgArray);
        #return hstack((imgArray, equali));
        return equali;

    @staticmethod
    def __convertToBinary(imgArray: ndarray, rescale: bool = False ) -> ndarray:
        assert(imgArray.ndim ==2), 'The image dimension is not too big'
        size: Tuple[int, int] = None;
        if(rescale):
            size = (150, 150)
        else:
            size = imgArray.shape
        img: Callable = resize(imgArray, size);
        return convertScaleAbs(img, alpha=1.10, beta=20);

    @staticmethod
    def histogram(imgArray: ndarray, preedit: bool = True) -> None:
        #print(imgArray.shape)
        hist(imgArray.ravel(), 256, [0,256]);
        if(preedit):
            title("Color Image Histogram");
        else:
            title("Black and White Image")
        show();
    @staticmethod
    def calculateHist(imgArray: ndarray, preedit:bool=True, output: bool = False) -> ndarray:
        if(preedit):
            channels: List[int] = [0]
        else:
            channels: List[int] = [0,1,2]
        hist: Callable = calcHist(imgArray, channels, None, [256], [0,256]);
        plot(hist)
        show()
        if(output):
            return asarray(hist)

    @staticmethod
    def showImage(imgArray: ndarray) -> None:
        imshow('Current Image', imgArray);
        waitKey(0);
        destroyAllWindows();


if __name__ == '__main__':

    imgName: str = r"snoopy.jpeg"
    pic: ndarray = imread(imgName);

    edition: ImageEditor = ImageEditor();
    #edition.histogram(pic)
    print(pic.shape)
    bwimage: ndarray = edition.editImagArray(pic, 'clahe')
    print(bwimage.shape)
    #edition.showImage(bwimage)
    bwimage1: ndarray = edition.editImagArray(pic, 'equalizationHist')
    print(bwimage1.shape)
    edition.showImage(bwimage1)
    bwimage2: ndarray = edition.editImagArray(pic, 'binary')
    print(bwimage2.shape)
    #edition.showImage(bwimage2)
    bwimage3: ndarray = edition.editImagArray(pic, None);
    print(bwimage3.shape)
    #edition.showImage(bwimage3)