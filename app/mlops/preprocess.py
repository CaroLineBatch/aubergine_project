import numpy as np
from PIL import Image
import glob
import io

#preprocess the image by reshaping it in 331*331
def preprocess(img):

    #get image size
    image = Image.open(io.BytesIO(img)).resize((331,331))
    width = image.width
    height = image.height
    print(width , height)
    data = []
    data.extend([np.array(image)])
    data = np.stack(data,axis=0)
    data =  np.round((data/255),3).copy()
    return data.reshape(-1,331, 331,3)


if __name__ == '__main__':
    img = glob.glob('/mnt/h/dataset-part1/dick/1qrv06cp0v471.jpg')
    print(preprocess(img))
