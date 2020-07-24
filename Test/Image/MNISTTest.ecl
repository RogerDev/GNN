/** MNISTTest.ecl is a test file to test the compatibility of each function with the MNIST dataset
  * Other than that, it also helps to check if the module is working properly for it's cause for MNIST operations
  * This test file may be used for a simple reference on how the flow must go when using GNN.Image with MNIST dataset
  */
IMPORT GNN.Image;
#option('outputLimit', 2000);

//Take MNIST dataset using Image module from suitable logical files
//MNIST train images
mnist_train_images := Image.MNIST.Get_train_images('~test::MNIST_train_images');
OUTPUT(mnist_train_images, NAMED('MNIST_train_images'));

//MNIST train labels
mnist_train_labels := Image.MNIST.Get_train_labels('~test::MNIST_train_labels');
OUTPUT(mnist_train_labels, NAMED('MNIST_train_labels'));

//MNIST test images
mnist_test_images := Image.MNIST.Get_test_images('~test::MNIST_test_images');
OUTPUT(mnist_test_images, NAMED('MNIST_test_images'));

//MNIST test labels
mnist_test_labels := Image.MNIST.Get_test_labels('~test::MNIST_test_labels');
OUTPUT(mnist_test_labels, NAMED('MNIST_test_labels'));

//Take first 25 records only, for test
testDs := mnist_train_images[..25];

//Make tensor for 25 mnist images
mnist_tens := Image.ImgtoTens(testDs);
OUTPUT(mnist_tens, NAMED('MNIST_tensor'));

//Take mnist images back from tensor
mnist_out := Image.TenstoImg(mnist_tens);
OUTPUT(mnist_out, NAMED('MNIST_out'));

//Output images as JPG
mnist_jpg := Image.OutputasJPG(mnist_out);
OUTPUT(mnist_jpg, ,'~test::mnist_jpg',OVERWRITE);
//PNG and BMP work the same way too!

//Output images as a Grid
mnist_grid := Image.OutputGrid(mnist_out, 5, 5, 'mnist_grid');
OUTPUT(mnist_grid, ,'~test::mnist_grid',OVERWRITE)