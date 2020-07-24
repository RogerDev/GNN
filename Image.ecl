IMPORT Python3 as Python;
IMPORT GNN.Tensor;
TensData := Tensor.R4.TensData;

/** ECL Image module
  * 
  * Prerequisites:
  * All the nodes using this module in the cluster must have the following python3 modules installed: -
  * - Python OpenCV
  * - Numpy
  * - Matplotlib
  * 
  * Overview:
  * This module tends towards the input, output and manipulation of images pertaining to neural network applications.  
  * This makes sure that the users of GNN do not spend time trying to preprocess the image database, 
  * as the functions in this module does most of the useful operations for them.  
  * 
  * We can divide the module as:
  * 1) Input
  * The module is capable of taking datasets of images sprayed as a blob, usually with the prefix: [filename,filesize]
  * This dataset sprayed is taken to obtain the image matrix so as to be sent to the neural network. 
  * Along with this, resizing or centered cropping of image is done so as to maintain uniformity in size for every image. 
  * This module also reads MNIST data from compressed unsigned byte files, in the submodule MNIST which is primarily for taking
  * input of MNIST dataset
  * 2) Output
  * The module takes raw byte data of the image from the Tensor output records and outputs the same as required. 
  * The user may use functions to output a dataset of images as a PNG or JPG or BMP or they can output the images 
  * as a grid in the form of a PNG image for checking multiple images conveniently.  
  * 3) Conversions
  * This module handles the preprocessing. It can convert records containing images as byte data into Tensor data 
  * to be able to use for conversion into a tensor and train the neural network using the tensor.  
  * It also handles the reverse transfer from Tensor data to image dataset to analyse outputs of the neural networks 
  *
  * Refer to the examples provided in the Test/Image folder to understand the working of each function and it's uses
  */
EXPORT Image := MODULE

  /** This record stores the Image labels for testing models with images. A label is 1 byte unsigned integer.
    * Mostly used for MNIST dataset.
    * @field id Index of the label, so that index of image may be matched
    * @field label Label stored as a number to be used conveniently
    */
  SHARED IMG_label := RECORD
    UNSIGNED id;
    UNSIGNED1 label;
  END;

  /** This record stores the images as per their index. Helps for maintaining the images and is simple to convert.
    * @field filename The name of the image as per the extracted dataset
    * @field image Image stored as an array of bytes which can be converted into integers or real values for using with neural networks.
    */  
  SHARED IMG_FORMAT := RECORD
    STRING filename;
    DATA image;
  END;

  /** This record stores the images as per their index. Helps for maintaining the images and is simple to convert.
    * @field id Index of the image which can also act as an identification factor for the image.
    * @field image Image stored as an array of bytes which can be converted into integers or real values for using with neural networks.
    * @field imgDims Dimensions of the image, that is, number of rows, columns and channels 
    * respectively used as a set of unsigned integers.
    */
  SHARED IMG_NUMERICAL := RECORD
    UNSIGNED8 id;
    DATA image;
    SET OF UNSIGNED imgDims;
  END;

  /** Separate module for MNIST data input
    * It has 4 functions, each of which work on separate 4 datasets by extracted ubyte files provided by Yann LeCunn.
    * The MNIST files for the following functions can be found in: http://yann.lecun.com/exdb/mnist/ 
    * Spray the following file as blob to be able to use the datasets
    */
  EXPORT MNIST := MODULE
    /** This function converts a compressed unsigned byte file, MNIST train images into a record for suitable input
      * It takes a logical file sprayed as a BLOB.  
      * Bytes are manipulated to convert the file into images.  
      * @param filename A string which would hold the filename for the logical file name or the landing zone file.  
      * @return A dataset of images from the file in the form of IMG_NUMERICAL record
      */
    EXPORT DATASET(IMG_NUMERICAL) Get_train_images(STRING filename) := FUNCTION
      //Record which is used to store the whole blob of MNIST train images
      MNIST_FORMAT := RECORD
          DATA4 magic;
          DATA4 numImages;
          DATA4 numRows;
          DATA4 numCols;
          DATA47040000 contents; //60000*28*28
      END;

      //Prerequisite data for the normalization
      mnist_imgs := DATASET(filename, MNIST_FORMAT, FLAT);
      numRows := (>UNSIGNED1<)mnist_imgs[1].numRows[4];
      numCols := (>UNSIGNED1<)mnist_imgs[1].numCols[4];
      imgSize := numRows*numCols;
      numImages := (>UNSIGNED2<) (mnist_imgs[1].numImages[4] + mnist_imgs[1].numImages[3]);

      //Normalization process where bytes of each image are split and put separately into IMG_NUMERICAL records
      mnistOut := NORMALIZE(mnist_imgs, numImages, TRANSFORM(IMG_NUMERICAL,
                                  SELF.image := LEFT.contents[((COUNTER-1)*imgSize+1) .. (COUNTER*imgSize)],
                                  SELF.id := COUNTER,
                                  SELF.imgDims := [numRows, numCols, 1]));
 
      RETURN mnistOut;                            
    END;

    /** This function converts a compressed unsigned byte file, MNIST test images into a record for suitable input
      * It takes a logical file sprayed as a BLOB.  
      * Bytes are manipulated to convert the file into images.
      * @param filename A string which would hold the filename for the logical file name or the landing zone file.  
      * @return A dataset of images from the file in the form of IMG_NUMERICAL record
      */
    EXPORT DATASET(IMG_NUMERICAL) Get_test_images(STRING filename) := FUNCTION
      //Record which is used to store the whole blob of MNIST test images
      MNIST_FORMAT := RECORD
          DATA4 magic;
          DATA4 numImages;
          DATA4 numRows;
          DATA4 numCols;
          DATA7840000 contents;
      END;

      //Prerequisite data for the normalization
      mnist_imgs := DATASET(filename, MNIST_FORMAT, FLAT);
      numRows := (>UNSIGNED1<)mnist_imgs[1].numRows[4];
      numCols := (>UNSIGNED1<)mnist_imgs[1].numCols[4];
      imgSize := numRows*numCols;
      numImages := (>UNSIGNED2<) (mnist_imgs[1].numImages[4] + mnist_imgs[1].numImages[3]);

      //Normalization process where bytes of each image are split and put separately into IMG_NUMERICAL records
      mnistOut := NORMALIZE(mnist_imgs, numImages, TRANSFORM(IMG_NUMERICAL,
                                  SELF.image := LEFT.contents[((COUNTER-1)*imgSize+1) .. (COUNTER*imgSize)],
                                  SELF.id := COUNTER,
                                  SELF.imgDims := [numRows, numCols, 1]));

      RETURN mnistOut;                            
    END;

    /** This function converts a compressed unsigned byte file, MNIST train labels into a record for suitable input
      * It takes a logical file sprayed as a BLOB. 
      * Bytes are manipulated to convert the file into labels.  
      * @param filename A string which would hold the filename for the logical file name or the landing zone file.  
      * @return A dataset of labels from the file in the form of IMG_label record
      */
    EXPORT DATASET(IMG_label) Get_train_labels(STRING filename) := FUNCTION
      //Record used to store the blob of MNIST train labels
      MNIST_FORMAT := RECORD
          DATA4 magic;
          DATA4 numImages;
          DATA60000 contents;
      END;

      //Prerequisite data for normalization
      mnist_lbls := DATASET(filename,MNIST_FORMAT,FLAT);
      numImages := (>UNSIGNED2<) (mnist_lbls[1].numImages[4] + mnist_lbls[1].numImages[3]);

      //Normalization process where bytes of labels are converted to unsigned, and put separately into IMG_label records
      mnistOut := NORMALIZE(mnist_lbls, numImages, TRANSFORM(IMG_label, 
                                          SELF.label := (>UNSIGNED1<)LEFT.contents[COUNTER],
                                          SELF.id := COUNTER;));

      RETURN mnistOut;
    END;

    /** This function converts a compressed unsigned byte file, MNIST test labels into a record for suitable input
      * It takes a logical file sprayed as a BLOB. 
      * Bytes are manipulated to convert the file into labels.  
      * @param filename A string which would hold the filename for the logical file name or the landing zone file.  
      * @return A dataset of labels from the file in the form of IMG_label record
      */
    EXPORT DATASET(IMG_label) Get_test_labels(STRING filename) := FUNCTION
      //Record used to store the blob of MNIST test labels
      MNIST_FORMAT := RECORD
          DATA4 magic;
          DATA4 numImages;
          DATA10000 contents;
      END;

      //Prerequisite data for normalization
      mnist_lbls := DATASET(filename,MNIST_FORMAT,FLAT);
      numImages := (>UNSIGNED2<) (mnist_lbls[1].numImages[4] + mnist_lbls[1].numImages[3]);

      //Normalization process where bytes of labels are converted to unsigned, and put separately into IMG_label records
      mnistOut := NORMALIZE(mnist_lbls, numImages, TRANSFORM(IMG_label, 
                                          SELF.label := (>UNSIGNED1<)LEFT.contents[COUNTER],
                                          SELF.id := COUNTER;));

      RETURN mnistOut;
    END;
  END;  //END of mnist module

  /** This function takes a logical file where multiple images may be sprayed as a blob with the prefix:[filename,filesize].
    * After obtaining dataset, converts into a dataset which numbers as indexes so as to traverse easier and making tensor
    * It also converts the input images to a different size given as a parameter, as explained below.
    * The metric of how the image is converted is also chosen using a parameter, as explained below. 
    * @param filename String containing the logical file of the image dataset
    * @param dims A set of unsigned numbers containing the dimensions of the output images as [number of rows, number of columns].
    * The channels need not be specified as the rest of the images are input by finding number of channels in the first image. 
    * That is, if the first image is grayscale, all other images even if in RGB are taken as grayscale and vice-versa.
    * This step is necessary so that the dataset converted into tensor is uniform for training of neural network.  
    * @param resize A Boolean variable which is defaulted to TRUE. 
    * If resize is true, resize of the image occurs as required which could result in squashed or stretched images
    * If resize is false, centering and cropping of image occurs. That is, only [numRows, numCols] shaped
    * image is taken about the centroid of the image obtained. 
    * Users may use any of these methods to maintain uniform data  
    * @return IMG_NUMERICAL dataset which can be converted to Tensor easily using a conversion function 
    */
  EXPORT DATASET(IMG_NUMERICAL) GetImages(STRING filename, SET OF UNSIGNED dims, BOOLEAN resize = TRUE) := FUNCTION
    /** Embedded Python function which reads image from the buffered DATA, decodes it suitably as per first image,
      * resizes or center crops the image and returns the obtained image matrix as bytearray to be returned as data
      * @param image DATA stream of the image given as input
      * @param dims Obtained set from the user to convert dimensions
      * @param color Whether color image or not. It is true if image is in color i.e. number of channels = 3. False for grayscale.
      * @param resize Whether to resize or not. Explained in above function documentation.  
      * @return DATA of obtained image with resize or centered crop in appropriate dimensions
      */
    DATA ReadImage(DATA image, SET OF UNSIGNED dims, BOOLEAN color, BOOLEAN resize) := EMBED(Python)
      import cv2
      import numpy as np
      image_np = np.frombuffer(image, dtype='uint8')
      if color:
        img = cv2.imdecode(image_np,cv2.IMREAD_COLOR)
      else:
        img = cv2.imdecode(image_np,cv2.IMREAD_GRAYSCALE)  
      dims = tuple(dims)
      if resize:
        img = cv2.resize(img, dims)
      else:
        y,x,_ = img.shape
        h,k = dims
        img = img[int((y-k)/2):int((y+k)/2), int((x-h)/2):int((x+h)/2)]
      return bytearray(img)
    ENDEMBED;

    /** Gets the image dimensions for checking number of channels of first image only
      * @param DATA stream of the image to get dimensions of
      * @return SET of integer that contains the shape of the image. 
      * It could be just an integer, but it's left so as it could help if there's a change or usecase for this function
      */
    SET OF INTEGER GetImageDimensions(DATA image) := EMBED(Python)
        import cv2
        import numpy as np 
        nparr = np.frombuffer(bytearray(image), dtype='uint8')
        img_np = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
        if len(img_np.shape) == 2:
          dims = list(img_np.shape)
          dims.append(1)
        else:
          dims = list(img_np.shape)  
        return dims;
    ENDEMBED;

    //Prerequisite data for the projection of image dataset
    imageData := DATASET(filename, IMG_FORMAT, FLAT);
    imgDims := dims + [GetImageDimensions(imageData[1].image)[3]];
    color := IF(imgDims[3] = 3, TRUE, FALSE);

    //Projection of image dataset to convert to iterable, uniform image dataset
    imageNumerical := PROJECT(imageData, TRANSFORM(IMG_NUMERICAL,
                                          SELF.id := COUNTER,
                                          SELF.image := ReadImage(LEFT.image, dims, color,resize),
                                          SELF.imgDims := imgDims));
    return imageNumerical;
  END;

  /** This function converts an IMG_Numerical record format to TensData format so that it can be passed into Tensor.R4.MakeTensor
    * This makes it easier to give image datasets as input to neural networks.
    * Images of any dimensions, grayscale or HSV or RGB are suitable for conversion in this.
    * The input dataset to this function is best obtained through ReadImages function defined above.
    * @param imgDataset The dataset of images which need to be converted to Tensor Data format.
    * @return Tensor data of the images which can be used to construct a t_Tensor using Tensor.R4.MakeTensor
    */
  EXPORT DATASET(TensData) ImgtoTens(DATASET(IMG_NUMERICAL) imgDataset) := FUNCTION
    //Prerequisities for the normalizaton into TensData
    imgShape := imgDataset[1].imgDims;
    imgRows := imgShape[1];
    imgCols := imgShape[2];
    imgChannels := imgShape[3];
    imgSize := imgRows * imgCols * imgChannels;

    /** Normalization using the obtained dimensions to convert appropriately to TensData
      * Changes each value to be in the range of -1 to 1 so as to be useful for neural network training
      */
    tens := NORMALIZE(imgDataset, imgSize, TRANSFORM(TensData,
                        SELF.indexes := [LEFT.id, (COUNTER-1) DIV (imgCols*imgChannels) +1, ((COUNTER-1) % (imgCols*imgChannels)) DIV imgChannels +1, (COUNTER-1) % imgChannels + 1],
                        SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 ));
    RETURN tens;                    
  END;

  /** This function converts the Tensor data output from the neural network into an Image dataset so it can be output in various forms
    * This is done so that the user can visualise and understand how successful their model has been. 
    * It converts a set of tensor values to bytes to achieve this.  
    * @param tens Tensor data that is needed to be converted to see as images
    * @return A dataset of IMG_NUMERICAL which can be output using below functions as required
    */
  EXPORT DATASET(IMG_NUMERICAL) TenstoImg(DATASET(TensData) tens) := FUNCTION
    /** As the name says, gives the bytes of the obtained image from a list of values so it can be obtained and used as image.  
      * @param input SET of unsigned which contains the values to be converted to bytes
      * @return byte array from python which is received as a block of data which is the image
      */
    DATA giveBytes(SET OF UNSIGNED input) := EMBED(Python)
        return bytearray(input)
    ENDEMBED;

    //Prerequisities for the Image dataset formation
    numImages := MAX(tens, tens.indexes[1]);
    numRows := MAX(tens, tens.indexes[2]);
    numCols := MAX(tens, tens.indexes[3]);
    numChannels := MAX(tens, tens.indexes[4]);

    /** Makes a dataset of images by converting the values back to the range of 0 to 255 for the image
      * First index is taken as identifier for each image. The values all images with the same 1st index are converted into a set. 
      * This set is sent to giveBytes to generate byte array for each image appropriately
      */
    imageDataset := DATASET(numImages,TRANSFORM(IMG_NUMERICAL,
                        SELF.id := COUNTER,
                        SELF.image := giveBytes(SET(tens(indexes[1]=COUNTER),(UNSIGNED)((value+1)*127.5))),
                        SELF.imgDims := [numRows, numCols, numChannels]));
    RETURN imageDataset;                    
  END;

  /** This function returns the image dataset with each image encoded into JPG images.  
    * When the output image dataset is desprayed with prefix: [filename,filesize], it returns images as jpg files.  
    * @param input Take the image dataset of input images to convert to JPG images
    * @param filename String parameter which lets the user make it easier to tag their output jpg images
    * The output would be of the form <id><filename>.jpg. It is defaulted as '', so if not specified 1.jpg, 2.jpg, etc. would be obtained
    * @return Image dataset having encoded JPG in bytearray of Images
    */
  EXPORT DATASET(IMG_FORMAT) OutputasJPG(DATASET(IMG_NUMERICAL) input, STRING filename = '') := FUNCTION
    /** Makes the JPG of each image in the dataset.  
      * Takes the image dataset with the dimensions of each image, reshape of the image is done to appropriate size,
      * image is encoded into type JPG and returned as array of bytes to be obtained as a JPG file, which can be desprayed
      * @param image Image that needs to be encoded
      * @return Returns the image in JPG encoded format
      */
    DATA makeJPG(DATA image, SET OF UNSIGNED dims) := EMBED(Python)
        import numpy as np
        import cv2
        image_np = np.frombuffer(image, dtype=np.uint8)
        image_np = image_np.reshape(dims)
        img_encode = cv2.imencode('.jpg', image_np)[1]
        return bytearray(img_encode)
    ENDEMBED;

    //Projection to perform the above function for each image
    output_jpg := PROJECT(input, TRANSFORM(IMG_FORMAT,
                        SELF.filename := LEFT.id + filename + '.jpg';
                        SELF.image := makeJPG(LEFT.image, LEFT.imgDims);
                        ));
    return output_jpg;                    
  END;

  /** This function returns the image dataset with each image encoded into PNG images.  
    * When the output image dataset is desprayed with prefix: [filename,filesize], it returns images as png files.  
    * @param input Take the image dataset of input images to convert to PNG images
    * @param filename String parameter which lets the user make it easier to tag their output png images
    * The output would be of the form <id><filename>.png. It is defaulted as '', so if not specified 1.png, 2.png, etc. would be obtained
    * @return Image dataset having encoded PNG in bytearray of Images
    */
  EXPORT DATASET(IMG_FORMAT) OutputasPNG(DATASET(IMG_NUMERICAL) input, STRING filename = '') := FUNCTION
    /** Makes the PNG of each image in the dataset.  
      * Takes the image dataset with the dimensions of each image, reshape of the image is done to appropriate size,
      * image is encoded into type PNG and returned as array of bytes to be obtained as a PNG file, which can be desprayed
      * @param image Image that needs to be encoded
      * @return Returns the image in PNG encoded format
      */
    DATA makePNG(DATA image, SET OF UNSIGNED dims) := EMBED(Python)
        import numpy as np
        import cv2
        image_np = np.frombuffer(image, dtype=np.uint8)
        image_np = image_np.reshape(dims)
        img_encode = cv2.imencode('.png', image_np)[1]
        return bytearray(img_encode)
    ENDEMBED;

    //Projection to perform the above function for each image
    output_png := PROJECT(input, TRANSFORM(IMG_FORMAT,
                        SELF.filename := LEFT.id + filename + '.png';
                        SELF.image := makePNG(LEFT.image, LEFT.imgDims);
                        ));
    return output_png;                    
  END;

  /** This function returns the image dataset with each image encoded into BMP images.  
    * When the output image dataset is desprayed with prefix: [filename,filesize], it returns images as BMP files.  
    * @param input Take the image dataset of input images to convert to BMP images
    * @param filename String parameter which lets the user make it easier to tag their output BMP images
    * The output would be of the form <id><filename>.bmp. It is defaulted as '', so if not specified 1.bmp, 2.bmp, etc. would be obtained
    * @return Image dataset having encoded BMP in bytearray of Images
    */
  EXPORT DATASET(IMG_FORMAT) OutputasBMP(DATASET(IMG_NUMERICAL) input, STRING filename = '') := FUNCTION
    /** Makes the BMP of each image in the dataset.  
      * Takes the image dataset with the dimensions of each image, reshape of the image is done to appropriate size,
      * image is encoded into type BMP and returned as array of bytes to be obtained as a BMP file, which can be desprayed
      * @param image Image that needs to be encoded
      * @return Returns the image in BMP encoded format
      */
    DATA makeBMP(DATA image, SET OF UNSIGNED dims) := EMBED(Python)
        import numpy as np
        import cv2
        image_np = np.frombuffer(image, dtype=np.uint8)
        image_np = image_np.reshape(dims)
        img_encode = cv2.imencode('.bmp', image_np)[1]
        return bytearray(img_encode)
    ENDEMBED;

    //Projection to perform the above function for each image
    output_bmp := PROJECT(input, TRANSFORM(IMG_FORMAT,
                        SELF.filename := LEFT.id + filename + '.bmp';
                        SELF.image := makeBMP(LEFT.image, LEFT.imgDims);
                        ));
    return output_bmp;                    
  END;

  /** This function converts any image dataset to an a grid of (r,c) where r*c should be size of the dataset.  
    * It makes a grid of (r,c) and encodes the image into a PNG to output. 
    * This helps for better understanding of the data which is obtained as output.  
    * The dataset output from this function may be desprayed with prefix: [filename,filesize] to obtain the PNG file
    * Note: Giving (1,1) for (r,c) is illegal and results in an error case. For the same, use OutputasPNG function instead.
    * @param Dataset of images to be converted to grid 
    * @param r Number of rows in the grid
    * @param c Number of columns in the grid
    * @param filename String which stores the filename to be given to the output image.
    * Output of the image would be a single record with filename as <filename>.png.
    * @return Dataset of 1 image which contains a PNG file with the resultant images in a grid
    */
  EXPORT DATASET(IMG_FORMAT) OutputGrid(DATASET(IMG_NUMERICAL) mnist, INTEGER r, INTEGER c, STRING filename) := FUNCTION
    /** Makes the grid using matplotlib module by treating each image given as a subplot
      * if (r,c) == (1,1), error is thrown as this does not need a grid
      * else if r == 1, the grid is generated simply by counting through columns
      * else if c == 1, the grid is generated simply by counting through rows
      * else, the grid is generated by traversing both directions and adding subplots perfectly
      * After adding subplots, these are drawn on the canvas in figure. The drawn image is taken as data and encoded as png.  
      * The encoded image is returned into the dataset.
      */
    DATA makeGrid(SET OF DATA images, Integer r, Integer c, SET OF UNSIGNED dims) := EMBED(Python)
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        fig, axs = plt.subplots(r, c)
        cnt = 0
        if dims[2] == 1:
          dims = dims[:2]
        if r == 1 and c == 1:
          assert 1==0, 'Error: Grid of (1,1) not possible. Use OutputasPNG function instead!'
        elif r == 1:
          for i in range(c):
            image = images[cnt]
            image_np = np.frombuffer(image, dtype=np.uint8)
            image_np = image_np.reshape(dims)
            axs[i].imshow(image_np[:,:], cmap='gray')
            axs[i].axis('off')
            cnt += 1
        elif c == 1:
          for i in range(r):
            image = images[cnt]
            image_np = np.frombuffer(image, dtype=np.uint8)
            image_np = image_np.reshape(dims)
            axs[i].imshow(image_np[:,:], cmap='gray')
            axs[i].axis('off')
            cnt += 1
        else:        
          for i in range(r):
            for j in range(c):
              image = images[cnt]
              image_np = np.frombuffer(image, dtype=np.uint8)
              image_np = image_np.reshape(dims)
              axs[i,j].imshow(image_np[:,:], cmap='gray')
              axs[i,j].axis('off')
              cnt += 1
        fig.canvas.draw()        
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
        plt.close()  
        img_encode = cv2.imencode('.png', image_from_plot)[1]
        return bytearray(img_encode)
    ENDEMBED;

    /** Generate dataset by making grid using above embedded function
      * All images are converted into a SET of DATA to be traversed through in the python function
      * Using the string filename provided, the name is given to the following image when desprayed
      */
    mnist_grid := DATASET(1, TRANSFORM(IMG_FORMAT,
                        SELF.filename := filename + '.png',
                        SELF.image := makeGrid(SET(mnist, image), r, c, mnist[1].ImgDims)
                        ));
    return mnist_grid;    
  END;
END; //End of Image module