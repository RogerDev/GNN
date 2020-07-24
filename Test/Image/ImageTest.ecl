/** ImageTest.ecl is to show how images may be read, converted and displayed using GNN.Image module
  * It was also used for testing the various functions in the module, for various conditions
  * This test file may be used as reference as to how the GNN.Image module may be used
  */
IMPORT GNN.Image;

//Input filename of dataset of images sprayed as logical file '~test::images_ds' as a blob with prefix: [filename,filesize]
//2nd parameter is shape of image to be used in the neural network
//3rd parameter if true resizes the image into the dimensions
ImageDs_resized := Image.GetImages('~test::images_ds', [100,100], TRUE);
OUTPUT(ImageDs_resized, NAMED('Input_images_resized'));

//3rd parameter if false, centers and crops the image into the dimensions
ImageDs_center_crop := Image.GetImages('~test::images_ds', [100,100], False);
OUTPUT(ImageDs_center_crop, NAMED('Input_images_centered_crop'));

//Convert obtained image dataset to tensors
ImageTens := Image.ImgtoTens(ImageDs_resized);
OUTPUT(ImageTens, NAMED('Images_tensor'));

//Convert tensor back to image dataset to output
OutputImageDs := Image.TenstoImg(ImageTens);
OUTPUT(OutputImageDs, NAMED('Output_Images'));

//Output the images as jpg files
OutputJpg := Image.OutputasJPG(OutputImageDs, '_test');
OUTPUT(Outputjpg, ,'~test::output_image_jpg',OVERWRITE);

//Output the images as png files
OutputPng := Image.OutputasPNG(OutputImageDs, '_test');
OUTPUT(OutputPng, ,'~test::output_image_png',OVERWRITE);

//Output the images as bmp files
OutputBmp := Image.OutputasBMP(OutputImageDs, '_test');
OUTPUT(Outputbmp, ,'~test::output_image_bmp',OVERWRITE);

//Rows and Columns in the grid. Make sure that gridRows*gridCols is the same as number of images given, or it will fail
//In the test, 4 images were used
gridRows := 2;
gridCols := 2;

//Output the image to despray as a grid
OutputGrid := Image.OutputGrid(OutputImageDs,gridRows,gridCols, 'test_out');
OUTPUT(OutputGrid, ,'~test::output_image_grid',OVERWRITE);

//Output input resized image as grid
OutputGridResized := Image.OutputGrid(ImageDs_resized,gridRows,gridCols, 'test_resized');
OUTPUT(OutputGridResized, ,'~test::input_resized_grid', OVERWRITE);

//Output input center cropped image as grid
OutputGridCenterCrop := Image.OutputGrid(ImageDs_center_crop,gridRows,gridCols, 'test_center_crop');
OUTPUT(OutputGridCenterCrop, ,'~test::input_center_crop_grid', OVERWRITE);