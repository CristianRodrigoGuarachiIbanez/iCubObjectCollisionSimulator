from PIL import Image
from os import listdir, getcwd
from os.path import join
from cython import locals
from numpy import ndarray, asarray
from typing import List, Dict
class ImageRetriever:

    def getDirectoryFiles(self, localDir: str ) -> ndarray:
        imgArrays: List[ndarray] = list()
        for strings in listdir(getcwd() + "/" + "{}".format(localDir)):
            if(strings.endswith(".png")):
                path: str = join(localDir, strings);
                print(strings)
                #print(self.__pngToArray(path)[0][0].shape)
                imgArrays.append(self.__pngToArray(path));
        return asarray(imgArrays);

    @staticmethod
    def __pngToArray(imgFile: str) -> ndarray:
        img: Image = Image.open(imgFile);
        return asarray(img);



if __name__ == '__main__':
    img: ImageRetriever = ImageRetriever();
    imgs: ndarray = img.getDirectoryFiles('image_outputs')
    print(imgs.shape)
    look: Image = Image.fromarray(imgs[0])
    look.show()


