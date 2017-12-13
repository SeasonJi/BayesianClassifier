#include <iostream>
#include <vector>
#include <map>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "bitmap.hpp"
#include <sstream>
#include <math.h>
#include <iomanip>
#define MNIST_DATA_DIR "../mnist_data"

using namespace std;
int main(int argc, char* argv[]) {
    //Read in the data set from the files
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_DIR);
    //Binarize the data set (so that pixels have values of either 0 or 1)
    mnist::binarize_dataset(dataset);
    //There are ten possible digits 0-9 (classes)
    int numLabels = 10;
    //There are 784 features (one per pixel in a 28x28 image)
    int numFeatures = 784;
    //Each pixel value can take on the value 0 or 1
    int numFeatureValues = 2;
    //image width
    int width = 28;
    //image height
    int height = 28;
    //image to print (these two images were randomly selected by me with no particular preference)
    int trainImageToPrint = 50;
    int testImageToPrint = 5434;
    // get training images
    std::vector<std::vector<unsigned char>> trainImages = dataset.training_images;//60000 train images
    // get training labels
    std::vector<unsigned char> trainLabels = dataset.training_labels;
    // get test images
    std::vector<std::vector<unsigned char>> testImages = dataset.test_images;//10000 test images
    // get test labels
    std::vector<unsigned char> testLabels = dataset.test_labels;



    int countEachDigit [] = {0,0,0,0,0,0,0,0,0,0};
    for(int i = 0; i < trainLabels.size(); i++)
    {
      int whichDigit = static_cast<int>(trainLabels[i]);//convert char to digit
      countEachDigit[ whichDigit ] ++ ;
    }

    //10 digits, each digit has 784 pixels
    //this 2d array count which pixel is white in a specific digit
    double eachDigitsPixel[10][784];
    for(int i = 0; i < 10; i++)
    {
      for(int j = 0; j < 784; j++)
      {
        //initialize with no white pixels counted
        eachDigitsPixel[i][j] = 0;
      }
    }

    //loop through all 60000 training images;
    //determine the which digit each image is;
    //count which pixel in each image is white;
    //and then update the corresponding count in the above 2d array;
    for (int i = 0; i < 60000; i++)//60000 images
    {
      int whichDigit = static_cast<int>(trainLabels[i]);
      for(int j = 0; j < 784; j++)//784 pixels in each image
      {
        int pixelValueAtThisIndex = static_cast<int>(trainImages[i][j]);
        if(pixelValueAtThisIndex == 1)//this pixel is white
        {
          eachDigitsPixel[whichDigit][j] ++ ;
        }
      }
    }

    //calculate probability that pixel Fj is white
    //given that it is an image of digit c
    double pixelDigitProbRaw [10][784];
    double pixelDigitProbProcessed [10][784];
    for(int i = 0; i < 10; i++)
    {
      int thisDigitCount = countEachDigit[i];
      for (int j = 0; j < 784; j++)
      {
        int whitePixelCount = eachDigitsPixel[i][j];
        pixelDigitProbRaw[i][j] = ((double)  (whitePixelCount)) / ((double) thisDigitCount);

        //When P(Fj = 1|C = c) = 0 (if pixel Fj has never been white in any
        //image of digit c), then, for any image in the test set of digit c where pixel Fj
        //is white, our classifier will predict that the probability that the image is of digit c is 0
        //To address this issue, we can pretend that we have observed every outcome
        //once more than we actually did (called Laplace smoothing): PL(Fj = 1|C = c),
        pixelDigitProbProcessed[i][j] = ((double) (whitePixelCount + 1)) / ( (double) (thisDigitCount + 2) ) ;
      }
    }

    //create bitmaps for digit 1 to 10
    for(int c = 0; c < numLabels; c++)
    {
      vector<unsigned char> classFs(numFeatures);
      for(int f = 0; f < numFeatures; f++)
      {
        double prob = pixelDigitProbProcessed[c][f];
        int v = 255 * prob;
        classFs[f] = (unsigned char)v;
      }
      stringstream ss;
      ss << "../output/digit" <<c<<".bmp";
      Bitmap::writeBitmap(classFs, 28, 28, ss.str(), false);
    }

    //creat network.txt
    //The first 784 lines should be P(Fj = 1|C = 0). The
    //next 784 lines should be P(Fj = 1|C = 1)
    //and so on for C=2(digit 2) to C=9(digit 9)
    ofstream file;
    file.open("../output/network.txt");
    for(int i = 0; i < numLabels; i++)
    {
      for(int j = 0; j < numFeatures; j++)
      {
        file << pixelDigitProbRaw[i][j] << endl;
      }
    }

    //network.txt's last ten lines are
    //prior probabilities for each calss(digit 0-9)
    for(int i = 0; i < numLabels; i++)
    {
      double prob = ((double)countEachDigit[i])/(60000.0);
      file << prob << endl;
    }
    file.close();


    //matrix for classification-summary.txt
    int predictMatrix [10][10];
    for(int i = 0; i < 10; i++)
    {
      for(int j = 0; j < 10; j++)
      {
        predictMatrix[i][j] = 0;//initialize this matrix
      }
    }

    //Now evaluate performance on testing set
    for(int testImgIndex = 0; testImgIndex < 10000; testImgIndex++)//totally 10000 images in testing set
    {
      int actualDigit = static_cast<int>(testLabels[testImgIndex]);
      int predictedDigit = -10000;
      double maxProb = INT_MIN;
      for(int i = 0; i < numLabels; i++)
      {
        double probSumForThisDigit = 0;
        for(int j = 0; j < numFeatures; j++)
        {
          int pixelValueAtThisIndex = static_cast<int>(testImages[testImgIndex][j]);
          double pixelClassProb = 0;

          if(pixelValueAtThisIndex == 1)
          {
            pixelClassProb = pixelDigitProbProcessed[i][j];
          }
          else if(pixelValueAtThisIndex == 0)
          {
            pixelClassProb = 1 - pixelDigitProbProcessed[i][j];
          }
          //add all the probabilituy together to
          //calculate the toatl probability of being digit i
          probSumForThisDigit += log2(pixelClassProb);

        }

        double probForThisDigitInAllTrainingImages  = ((double)countEachDigit[i]) / 60000.0;
        probSumForThisDigit += log2(probForThisDigitInAllTrainingImages);
        if(probSumForThisDigit > maxProb)
        {
          maxProb = probSumForThisDigit;
          predictedDigit = i; //our algorithm predicts that this image is digit i
        }
      }
      //update matrix value
      predictMatrix[actualDigit][predictedDigit]++;

    }

    ofstream ofile;
    ofile.open("../output/classification-summary.txt");
    int correctPredictionsCount = 0;
    for(int i = 0; i < 10; i++)
    {
      for(int j = 0; j < 10; j++)
      {
        ofile << setw(6) << predictMatrix[i][j] << " ";
        if(i == j)//correct prediction!
        {
          correctPredictionsCount += predictMatrix[i][j];
        }
      }
      ofile << endl;
    }

    double accuracy = (((double)correctPredictionsCount) / 10000.0) * 100;
    ofile << accuracy << "%" << endl;
    ofile.close();



    return 0;


    // //print out one of the training images
    // for (int f=0; f<numFeatures; f++) {
    //     // get value of pixel f (0 or 1) from training image trainImageToPrint
    //     int pixelIntValue = static_cast<int>(trainImages[trainImageToPrint][f]);
    //     if (f % width == 0) {
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<pixelIntValue<<" ";
    // }
    // std::cout<<std::endl;
    // // print the associated label (correct digit) for training image trainImageToPrint
    // std::cout<<"Label: "<<static_cast<int>(trainLabels[trainImageToPrint])<<std::endl;
    // //print out one of the test images
    // for (int f=0; f<numFeatures; f++) {
    //     // get value of pixel f (0 or 1) from training image trainImageToPrint
    //     int pixelIntValue = static_cast<int>(testImages[testImageToPrint][f]);
    //     if (f % width == 0) {
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<pixelIntValue<<" ";
    // }
    // std::cout<<std::endl;
    // // print the associated label (correct digit) for test image testImageToPrint
    // std::cout<<"Label: "<<static_cast<int>(testLabels[testImageToPrint])<<std::endl;
    // std::vector<unsigned char> trainI(numFeatures);
    // std::vector<unsigned char> testI(numFeatures);
    // for (int f=0; f<numFeatures; f++) {
    //     int trainV = 255*(static_cast<int>(trainImages[trainImageToPrint][f]));
    //     int testV = 255*(static_cast<int>(testImages[testImageToPrint][f]));
    //     trainI[f] = static_cast<unsigned char>(trainV);
    //     testI[f] = static_cast<unsigned char>(testV);
    // }
    // std::stringstream ssTrain;
    // std::stringstream ssTest;
    // ssTrain << "../output/train" <<trainImageToPrint<<"Label"<<static_cast<int>(trainLabels[trainImageToPrint])<<".bmp";
    // ssTest << "../output/test" <<testImageToPrint<<"Label"<<static_cast<int>(testLabels[testImageToPrint])<<".bmp";
    // Bitmap::writeBitmap(trainI, 28, 28, ssTrain.str(), false);
    // Bitmap::writeBitmap(testI, 28, 28, ssTest.str(), false);
    // return 0;
}
