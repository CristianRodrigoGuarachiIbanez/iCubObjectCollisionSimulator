from numpy import ndarray, array, asarray, append, round, hstack, mean, newaxis, reshape
from matplotlib.pyplot import hist, title, show, plot
from cv2 import imread, imshow, waitKey, destroyAllWindows,calcHist, resize, convertScaleAbs, equalizeHist, createCLAHE, \
    cvtColor,  COLOR_BGR2GRAY, THRESH_BINARY, threshold
from typing import List, Tuple, Callable

class ImageEditor:

    def editImagArray(self, imgArray: ndarray, equa_method: str, scale: int = 30):
        '''
        convert a Img with shape (None, 230, 270, 3) into a gray scale img with (None, 120, 160)  and a equalizes the image with a shape transforming it into
        a img with shape (None, 120,160,1)
        '''
        #bwimage: ndarray = self.__resizeImage(self.__convertImgToBW(imgArray=imgArray), scale_percent=scale)
        grayImg: ndarray = self.__resizeImage( self.__convertImgToGS(imgArray=imgArray), scale_percent=scale)
        #print(grayImg.shape)
        if(equa_method == 'clahe'):
            return self.addNewChannel(self.__claheEqualization(grayImg), new_axis=True);
        elif(equa_method=='equalizationHist'):
            return self.__equalizeHistagram(grayImg);
        elif(equa_method=='binary'):
            return self.__convertToBinary(grayImg);
        elif(equa_method=='gray'):
            return grayImg

    @staticmethod
    def addNewChannel(imgArray: ndarray, channel: int = 1, new_axis: bool = False) -> ndarray:
        if (new_axis):
            return imgArray[..., newaxis];
        else:
            if(imgArray.shape==4):
                return imgArray.reshape((imgArray.shape[0], imgArray.shape[1], imgArray.shape[2], imgArray.shape[3],  channel))
            elif(imgArray.shape==3):
                return imgArray.reshape((imgArray.shape[0], imgArray.shape[1], imgArray.shape[2],  channel))
            elif(imgArray.shape==2):
                return imgArray.reshape((imgArray.shape[0], imgArray.shape[1], channel))

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
    def showImage(imgArray: ndarray, index: int) -> None:
        imshow('Current Image'+ '{}'.format(index), imgArray);
        waitKey(200000);
        #destroyAllWindows();


if __name__ == '__main__':

    pass
