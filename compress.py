from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_name):
    im = Image.open(image_name)
    return np.array(im)

def save_image(A,image_name):
    im = Image.fromarray(A.astype('uint8'))
    im.save(image_name)
    
class Img_copresator_same_eig:
    def __init__(self,high,width):
        self.high = high
        self.width = width
        
    def fit(self,image):
        self.image_shape = image.shape
        X = image.reshape(-1,self.high*self.width*3)
        self.mean = np.mean(X,axis=0)
        cov_X = np.cov(X.T)
        w, v = np.linalg.eigh(cov_X)
        idx = w.argsort()[::-1]
        self.eigen_values = w[idx]
        self.eigen_vectors = v[:,idx]
    def code(self,image,dim):
        X = image.reshape(-1,self.high*self.width*3)
        X = (X - self.mean)

        Trans = self.eigen_vectors[:,0:dim]
        X_trans= np.dot(X,Trans)
        return X_trans
        
        
    def decode(self,code,dim):
        Trans = (self.eigen_vectors[:,0:dim]).T
        X = np.dot(code,Trans) + self.mean
        return X.reshape(self.image_shape) 
    def copressed_size(self,image,dim):
        return (dim + 1)* self.high * self.width * 3 + len((self.code(image,dim)).reshape(-1))
       
class Img_compresator:
    def __init__(self,high_sub_img,width_sub_img,high,width):
        self.high = high
        self.width = width
        self.high_sub_img =  high_sub_img
        self.width_sub_img = width_sub_img
    
    def fit(self,image):
        self.image_shape = image.shape
        X=image.reshape(-1,self.high_sub_img,self.high_sub_img,3)
        self.subimgcompresators = [Img_copresator_same_eig(self.high,self.width) for i in range(len(X))]
        for i in range(len(X)):
            self.subimgcompresators[i].fit(X[i])
    def code(self,image,dim):
        X = image.reshape(-1,self.high_sub_img,self.high_sub_img,3)
        code = np.array([self.subimgcompresators[i].code(X[i],dim) for i in range(len(X))])
        return code
    def decode(self,code,dim):
        X = np.array([self.subimgcompresators[i].decode(code[i],dim) for i in range(len(code))])
        return X.reshape(self.image_shape)
    
    def copressed_size(self,image,dim):
        X = image.reshape(-1,self.high_sub_img,self.high_sub_img,3)
        size = 0;
        for i in range(len(X)):
            size +=self.subimgcompresators[i].copressed_size(X[i],dim)
        return size
        
    
def Algorytm(im,d,high1,width1,high2,width2,name,save = False):
    """@im image
       @d dimension of compresion 
       @high1 high of subimages in original image
       @width1 width of subimages in original image
       @high2 high of subimage in subimages
       @width2 width of subimage in subimages
       @name name used to create copresed image file
       @save if True image is saved
        
    
    """
    img_copresator = Img_compresator(high1,width1,high2,width2)
    img_copresator.fit(im)
    code_im = img_copresator.code(im,d)
    im_decoded = img_copresator.decode(code_im,d)
    im_decoded = np.array(im_decoded, dtype=np.uint8)
    if save == True:
        save_image(im_decoded.astype(int),name+"I"+str(high1)+"x"+str(width1)+"I"+str(high2)+"x"+str(width2)+"I"+str(d)+".bmp")
    return im_decoded,img_copresator  
