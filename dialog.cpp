#include "dialog.h"
#include "ui_dialog.h"
#include "imageitem.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/face.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <QList>
#include <QHBoxLayout>
#include <QMenu>
#include <QString>
#include <QTimer>
#include <string>


#define MAX_NAME 100
#define NAME_FORMAT "%1.pgm"
#define DIR_NAME_FORMAT "s%1"

using namespace cv;
using namespace std;
using namespace cv::face;


static cv::Mat norm_0_255(InputArray _src) {
    cv::Mat src = _src.getMat();
    // Create and return normalized image:
    cv::Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

static void read_csv(const string& filename, vector<cv::Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(cv::Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        std::stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            cv::Mat imagetemp = imread(path, 0);
            images.push_back(imagetemp);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

static void write_csv(const string& filename, vector<int>& in_labels, vector<string>& in_files, char separator = ';') {
    std::ofstream file;
    file.open(filename.c_str(), ios::app);
    for( int i = 0; i < in_labels.size(), i < in_files.size(); ++i)
    {
        file << in_files.at(i) << separator << in_labels.at(i) << std::endl;
    }
    file.close();
}


static bool fileExists(QString file)
{
    QFileInfo checkFile(file);
    if (checkFile.exists() && checkFile.isFile()) {
        return true;
    } else  {
        return false;
    }
}
static bool dirExists(QString dir)
{
    QFileInfo checkFile(dir);
    if (checkFile.exists() && checkFile.isDir()) {
        return true;
    } else  {
        return false;
    }
}
static int nextName(QString folder)
{
    for( int i = 1; i <= MAX_NAME; ++i)
    {
        QString searchFile = QString(folder + "/" + NAME_FORMAT).arg(i);
        if(!fileExists(searchFile))
        {
            return i;
        }
    }
    return 0;
}
static int nextDirName(QString folder)
{
    for( int i = 1; i <= MAX_NAME; ++i)
    {
        QString searchFile = QString(folder + "/" + DIR_NAME_FORMAT).arg(i);
        if(!dirExists(searchFile))
        {
            return i;
        }
    }
    return 0;
}

static cv::Mat scaleImage(cv::Mat in_src, cv::Mat dest, double destX, double destY, bool iscvtclr = true)
{
    cv::Mat ret = dest.clone();
    cv::Mat src;
    cv::Mat destgray = cv::Mat(dest.rows,dest.cols, dest.depth());
    if(iscvtclr)
    {
        cvtColor(in_src, src, cv::COLOR_BGR2GRAY);
    }
    else
    {
        src = in_src;
    }

    double perX =  destX / src.cols;
    double perY = destY / src.rows;
    double scale;

    int diffx1;
    int diffx2;
    int diffy1;
    int diffy2;

    // scale y but find the width
    // given original is larger, the smaller the # the more difference
    if(perX < perY)
    {
        // scale by x
        scale = perY;
        // amount to remove from the sides

        diffx1 = ceil((src.cols * scale) - destX) / 2;
        diffx2 = floor((src.cols * scale) - destX) / 2;
        diffy1 = 0;
        diffy2 = 0;

        // difference in widths
        // need to be able to get difference in x between: the original scaled and destination

        // but the ratio of the y scale, and

        // ratio of difference
        //
    }
    else
    {
        scale = perX;

        // amount to remove from the sides
        diffx1 = 0;
        diffx2 = 0;
        diffy1 = ceil((src.rows * scale) - destY) / 2;
        diffy2 = floor((src.rows * scale) - destY) / 2;
    }


    // ratio of: x to

//    xScale = perY;
//    yScale = perY;

    // scale x and y to the y and

    // how much destination x is greater than

    int xScaledDown = (int)src.cols * scale;
    int yScaledDown = (int)src.rows * scale;

    // get new size without cropping
    Mat matScaled2 = Mat(yScaledDown, xScaledDown , dest.type());

    cv::resize(src, matScaled2, cv::Size(xScaledDown ,yScaledDown));


//    QImage qimgProcessed((uchar*)matScaled2.data, matScaled2.cols, matScaled2.rows, matScaled2.step, QImage::Format_Indexed8);
//    ui->lblOriginal->setPixmap(QPixmap::fromImage(qimgProcessed));
//    return;

    //matScaled2.copyTo(originalSize(Rect(0, 0, yScaledDown, xScaledDown )));

    // difference between desired and diffx


    // xRemain = diffx + xScaledDown
    //

    // desired size subtract


//    int nameIter2 = nextName(QString::fromStdString(output_folder));
//    if(nameIter2 == 0) {
//        return;
//    }
//    QString outFileName2 = QString(QString::fromStdString(output_folder) + "/%1.pgm").arg(nameIter2);
//    imwrite(outFileName2.toStdString(), originalSize);

    cv::Rect roi( diffx1, diffy1, xScaledDown - diffx2 - diffx1, yScaledDown - diffy2 - diffy1 );
    cv::Mat matScaledClipped = matScaled2( roi );

    matScaledClipped.copyTo( ret );

    return ret;
}

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    try
    {
        imgCount = 0;
        selImagePos = -1;
        selimage = 0;
        ui->setupUi(this);

        model = cv::face::FisherFaceRecognizer::create();

        QFileInfo checkFile(modelfilepath);
        if(checkFile.exists())
        {
            model->read(  modelfilepath.toStdString() );
        }



        imagePreviewLayout = ui->imageItemsLayout;
        facerecProgress = ui->progressBar;
        facerecProgress->setValue(0);

        sizePolicyFixed = QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicyExpand = QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

        fn_csv = "/home/devon/FaceRecognition/faces1.csv";
        output_folder = "/home/devon/FaceRecognition/att_faces";

        ui->txtCsvInputPath->setText(QString::fromStdString(fn_csv));
        ui->txtImgOutputPath->setText(QString::fromStdString(output_folder));

//        imageItem* imgAddSpacer = new imageItem(imagePreviewLayout->widget());
//        imgAddSpacer->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
//        imagePreviewLayout->addWidget(imgAddSpacer);

        capWebcam.open(0);

        if(capWebcam.isOpened() == false)
        {

        }

        //double fps = capWebcam.get(CV_CAP_PROP_FPS);

//        String face_cascade_namecvstr2 = "smoething";
//        if( !face_cascade->load( face_cascade_namecvstr2 ) ){ /*ui->txtOutput->append(QString::fromStdString("--(!)Error loading faces xml\n")); */}
//        if( !eyes_cascade->load( String(eyes_cascade_name.toStdString().c_str()) ) ){ ui->txtOutput->append(QString::fromStdString("--(!)Error loading eyes xml\n")); }

        //cv::Mat src_host = cv::imread(vidFile.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
        //cv::gpu::GpuMat dst, src;

        reloadImages();


        //use the haarcascade_frontalface_alt.xml library
        face_cascade_run = cv::cuda::CascadeClassifier::create(face_cascade_name.toStdString());
        eyes_cascade_run = cv::cuda::CascadeClassifier::create(eyes_cascade_name.toStdString());

        tmrTimer = new QTimer(this);
        connect(tmrTimer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));
        tmrTimer->start(1);

    }
    catch(const cv::Exception& ex)
    {
        QMessageBox msgBox;
        msgBox.setText(ex.what());
        msgBox.exec();
    }
}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::on_btnImagePath_clicked()
{
    QString qFilePath = QFileDialog::getExistingDirectory(this, tr("Select Image Output Path"), "", QFileDialog::ShowDirsOnly);
    output_folder = qFilePath.toStdString();
    ui->txtImgOutputPath->setText(qFilePath);
}

void Dialog::on_btnCsvFile_clicked()
{
    QString qCsvFilePath = QFileDialog::getOpenFileName(this, tr("Select Input CSV File"), "", tr("All files (*.*)"));
    fn_csv = qCsvFilePath.toStdString();
    ui->txtCsvInputPath->setText(qCsvFilePath);
}


void Dialog::processFrameAndUpdateGUI()
{
    capWebcam.read(matOriginal2);
    if(matOriginal2.empty() == true) return;

    //QImage qimgProcessed2((uchar*)matOriginal2.data, matOriginal2.cols, matOriginal2.rows, matOriginal2.step, QImage::Format_RGB888);
//    ui->lblFeed->setPixmap(QPixmap::fromImage(qimgProcessed2));

     detectAndDisplay( matOriginal2 );
//     cannyAndDisplay(matOriginal2);

}
void Dialog::cannyAndDisplay(cv::Mat camMat) {

}

void Dialog::on_btnFindFaces_clicked()
{
    try
    {
        ui->txtOutput->clear();

        cv::Mat snapFirst;
        cv::cuda::GpuMat snapImage, snapImageGray, snapImageGray2;
        cv::Mat matPreScaler, matScaled;

        int pad = 40;

        QString sample_folder = QString::fromStdString(output_folder) + "/output";
        if(tmrTimer->isActive() == true) {
            tmrTimer->stop();
        }

        if( !fileExists(QString::fromStdString(fn_csv) )) {
            return;
        }

        int nameIter = nextName(sample_folder);
        if(nameIter == 0) {
            return;
        }

        QString outFileName = QString(sample_folder + "/%1.pgm").arg(nameIter);



        capWebcam.read(snapFirst);
        if(snapFirst.empty() == true) return;


        snapImage.upload(snapFirst);

         //Debug point
        //cv::Mat faceROIout;

        //snapImage.download(faceROIout);



//        QImage qimgProcessed((uchar*)faceROIout.data, faceROIout.cols, faceROIout.rows, faceROIout.step, QImage::Format_RGB888);

//        ui->lblOriginal->setPixmap(QPixmap::fromImage(qimgProcessed));



        // using gpu face detection


        //gpuMatOriginal.upload(frame);


        cv::cuda::cvtColor( snapImage, snapImageGray, cv::COLOR_BGR2GRAY );

//        cv::cvtColor( snapImage, snapImageGray, CV_BGR2GRAY );

        cv::cuda::equalizeHist( snapImageGray,  snapImageGray2);

        cv::cuda::GpuMat objbuf;


        // unused
//        int minNeighborsFace = 2;

        std::vector<Rect> faces;


        face_cascade_run->detectMultiScale( snapImageGray2, objbuf/*, 1.2, minNeighborsFace, Size(30,30)*/);

        face_cascade_run->convert(objbuf, faces);

//        Mat obj_host(objbuf);
        // download only detected number of rectangles


        //for(int i = 0; i < detections_num; ++i)
      //     cv::rectangle(image_cpu, faces[i], Scalar(255));







        for( size_t i = 0; i < faces.size(); i++ )
        {
            Point pt1(faces[i].x + faces[i].width + pad, faces[i].y + faces[i].height + pad);
            Point pt2(faces[i].x - pad, faces[i].y - pad);

            cv::Rect roi (pt1, pt2);

            cv::cuda::GpuMat faceROI = snapImageGray2( roi );

            Mat originalSize = imread("/home/devon/FaceRecognition/sizer.pgm", 0);

//            cv::Mat faceROIout;

            //Debug output

//            faceROI.download(faceROIout);

//            QImage qimgProcessed((uchar*)faceROIout.data, faceROIout.cols, faceROIout.rows, faceROIout.step, QImage::Format_RGB888);

//            ui->lblOriginal->setPixmap(QPixmap::fromImage(qimgProcessed));

            faceROI.download(matPreScaler);





            //int height = images[0].rows;



            matScaled = scaleImage(matPreScaler, originalSize, 92.0, 112.0, false);




            Mat testSample = norm_0_255(matScaled.reshape(1, originalSize.rows));
            int testLabel = nextDirName(QString::fromStdString(output_folder));


            QImage qimgProcessed((uchar*)testSample.data, testSample.cols, testSample.rows, testSample.step, QImage::Format_RGB888);

            ui->lblOriginal->setPixmap(QPixmap::fromImage(qimgProcessed));


            int predictedLabel = model->predict(testSample);
            //
            // To get the confidence of a prediction call the model with:
            //
            //      int predictedLabel = -1;
            //      double confidence = 0.0;
            //      model->predict(testSample, predictedLabel, confidence);
            //




            string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
            ui->txtOutput->append(QString::fromStdString(result_message));
//            //cout << result_message << endl;
//            // Here is how to get the eigenvalues of this Eigenfaces model:
//            Mat eigenvalues = model->getMat("eigenvalues");
//            // And we can do the same to display the Eigenvectors (read Eigenfaces):
//            Mat W = model->getMat("eigenvectors");
//            // Get the sample mean from the training data
//            Mat mean = model->getMat("mean");
            // Display or save:




//            imwrite(QString("%1/mean.png").arg(sample_folder).toStdString(), norm_0_255(mean.reshape(1, 112)));

//            int procCount = 0;
//            facerecProgress->setValue(0);
//            facerecProgress->setRange(0, min(16, W.cols) * 2);





//            // Display or save the first, at most 16 Fisherfaces:
//            for (int i = 0; i < min(16, W.cols); i++) {

//                string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
//                ui->txtOutput->append(QString::fromStdString(msg));
//                //cout << msg << endl;
//                // get eigenvector #i
//                Mat ev = W.col(i).clone();
//                // Reshape to original size & normalize to [0...255] for imshow.
//                Mat grayscale = norm_0_255(ev.reshape(1, 112));
//                // Show the image & apply a Bone colormap for better sensing.
//                Mat cgrayscale;
//                applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
//                // Display or save:

//                imwrite(QString("%1/fisherface_%2.png").arg(sample_folder).arg(i).toStdString(), norm_0_255(cgrayscale));
//                facerecProgress->setValue(++procCount);
//            }


//            // Display or save the image reconstruction at some predefined steps:
//            for(int num_component = 0; num_component < min(16, W.cols); num_component++) {

//                // Slice the Fisherface from the model:
//                Mat ev = W.col(num_component);
//                Mat projection = subspaceProject(ev, mean, images[0].reshape(1,1));
//                Mat reconstruction = subspaceReconstruct(ev, mean, projection);
//                // Normalize the result:
//                reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
//                // Display or save:

//                //imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
//                imwrite(QString("%1/fisherface_reconstruction_%2.png").arg(sample_folder).arg(num_component).toStdString(), reconstruction);
//                facerecProgress->setValue(++procCount);

//            }

        }

    } catch (cv::Exception& e) {

        ui->txtOutput->append(QString::fromStdString("Error: " + e.msg));
        //cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        tmrTimer->start(1);
        return;

    }


    ui->txtOutput->append(QString::fromStdString("Done!"));
    // Display if we are not writing to an output folder:
    tmrTimer->start(1);

    return;
}


void Dialog::clearLayout(QLayout* layout /*QWidget* widget*/, bool deleteWidgets)
{

    if (layout != 0)
    {
        while (QLayoutItem* item = layout->takeAt(0))
        {
            if (deleteWidgets)
            {
                if (QWidget* widget = item->widget())
                    delete widget;
            }
            if (QLayout* childLayout = item->layout())
                clearLayout(childLayout, deleteWidgets);
            delete item;
        }
    }
    //delete layout;
}

void Dialog::on_btnSaveSnap_clicked()
{
    //create the cascade classifier object used for the face detection
    CascadeClassifier face_cascade;
    //use the haarcascade_frontalface_alt.xml library
    face_cascade.load("/home/devon/FaceRecognition/haarcascade_frontalface_alt.xml");

    capWebcam.read(matOriginal3);
    if(matOriginal3.empty() == true) return;

    Mat grayMatOriginal3;
    Mat histMat3;
    // Convert to grayscale
    cvtColor( matOriginal3, grayMatOriginal3, cv::COLOR_BGR2GRAY );

    // Apply Histogram Equalization
    equalizeHist( grayMatOriginal3, histMat3 );

    std::vector<Rect> faces;

    //find faces and store them in the vector array
    face_cascade.detectMultiScale(histMat3, faces, 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT|CASCADE_SCALE_IMAGE , Size(30,30));
    int pad = 40;
    //draw a rectangle for all found faces in the vector array on the original image
    for(int i = 0; i < faces.size(); i++)
    {
        Point pt1(faces[i].x + faces[i].width + pad, faces[i].y + faces[i].height + pad);
        Point pt2(faces[i].x - pad, faces[i].y - pad);

        cv::Rect roi (pt1, pt2);
        cv::Mat matOriginalFocus3 = matOriginal3( roi );

        // mutli person image
        //imshow("outputCapture", matOriginalFocus3);

        Mat originalSize = imread("/home/devon/FaceRecognition/sizer.pgm", 0);

        originalSize3 = scaleImage(matOriginalFocus3, originalSize, 92.0, 112.0);

        imgSnapList.push_back(originalSize3);

//        QImage qimgProcessed((uchar*)originalSize3.data, originalSize3.cols, originalSize3.rows, originalSize3.step, QImage::Format_Indexed8);
        // create
//        imageItem* imgAdd = new imageItem(this);
//        imgAdd->pos = imgCount;

//        imgAdd->setPixmap(QPixmap::fromImage(qimgProcessed));
//        //edit
//        imgGridLayout->addWidget((QWidget*)imgAdd, 0, imgCount++);
//        imgSnapWidgets.append(imgAdd);

//        imgAdd->setContextMenuPolicy(Qt::CustomContextMenu);
//        connect(imgAdd, SIGNAL(customContextMenuRequested(const QPoint&)),
//            this, SLOT(ShowContextMenu(const QPoint&)));


//        QImage qimgProcessed((uchar*)originalSize3.data, originalSize3.cols, originalSize3.rows, originalSize3.step, QImage::Format_Indexed8);

//        imageItem* imgAdd = new imageItem(imagePreviewLayout->widget());
//        imgAdd->pos = imgCount;
//        imgAdd->setGeometry(QRect(0,0, 92, 112));
//        imgAdd->setFixedWidth(92);
//        imgAdd->setFixedHeight(112);
//        imgAdd->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));

//        imgAdd->setPixmap(QPixmap::fromImage(qimgProcessed));
//        //edit
//        imagePreviewLayout->addWidget(imgAdd);
//        imgSnapWidgets.append(imgAdd);

//        imgAdd->setContextMenuPolicy(Qt::CustomContextMenu);
//        connect(imgAdd, SIGNAL(customContextMenuRequested(const QPoint&)),
//        this, SLOT(ShowContextMenu(const QPoint&)));

        //++imgCount;
    }
    reloadImages();


//    imshow("outputCapture", matOriginalFocus3);

//    Mat originalSize = imread("E:/Source/QtCVFaces/sizer.pgm", 0);

//    originalSize3 = scaleImage(matOriginal3, originalSize, 92.0, 112.0);

//    imgSnapList.push_back(originalSize3);

//    QImage qimgProcessed((uchar*)originalSize3.data, originalSize3.cols, originalSize3.rows, originalSize3.step, QImage::Format_Indexed8);
//    // create
//    QLabel* imgAdd = new QLabel("", this);
//    imgAdd->setPixmap(QPixmap::fromImage(qimgProcessed));
//    imgGridLayout->addWidget(imgAdd, 0, imgCount++);
}
void Dialog::changeSelImage()
{
    if(selimage)
    {
        selimage->setStyle("QLabel { background: none; }");
    }
    selimage = (imageItem*)qApp->widgetAt(QCursor::pos());
    if(selimage)
    {
        selimage->setStyle("QLabel { background: blue; } ");
    }
    return;

}

void Dialog::reloadImages()
{
    imgSnapWidgets.clear();

    if(imagePreviewLayout != 0)
    {
        clearLayout(imagePreviewLayout->layout());
    }

    imageItem* imgAddSpacer = new imageItem(this);
    imgAddSpacer->setSizePolicy(sizePolicyExpand);
    imgAddSpacer->setContentsMargins(5,5,5,5);
    imagePreviewLayout->addWidget(imgAddSpacer);

    imgCount = 0;



    for( QList<cv::Mat>::iterator iter = imgSnapList.begin(); iter < imgSnapList.end(); ++iter)
    {
        cv::Mat originalSize4 = (*iter);
        QImage qimgProcessed((uchar*)originalSize4.data, originalSize4.cols, originalSize4.rows, originalSize4.step, QImage::Format_Indexed8);

        imageItem* imgAdd = new imageItem(this);
        imgAdd->pos = imgCount++;
        imgAdd->setFixedWidth(92);
        imgAdd->setFixedHeight(112);
        imgAdd->setContentsMargins(5,5,5,5);
        //imgAdd->setSizePolicy(sizePolicyFixed);

        imgAdd->setPixmap(QPixmap::fromImage(qimgProcessed));


        imagePreviewLayout->addWidget(imgAdd);

        imgSnapWidgets.append(imgAdd);

        connect(imgAdd, SIGNAL(clicked()), this, SLOT(changeSelImage()));

    }
}

void Dialog::removeImage()
{
    if (selImagePos != -1)
    {
        imgSnapList.removeAt(selImagePos);
        reloadImages();
        selImagePos = -1;
        return;
    }
}


void Dialog::on_btnExportFaces_clicked()
{
    std::vector<int> outImageNumbers;
    std::vector<std::string> outImageNames;

    int dirNameIter = nextDirName(QString::fromStdString(output_folder));
    QString newDirPath = QString(QString::fromStdString(output_folder) + "/s%1").arg(dirNameIter);

    if(dirNameIter == 0 ){
       // throw exception
       return;
    }
    QDir().mkdir(newDirPath);

    for( QList<cv::Mat>::iterator iter = imgSnapList.begin(); iter < imgSnapList.end(); ++iter)
    {
        int fileNameIter = nextName(newDirPath);
        if(fileNameIter == 0)
        {
            return;
        }
        QString newFilePath = QString(newDirPath + "/%1.pgm").arg(fileNameIter);
        // create images in path
        imwrite(newFilePath.toStdString(), (*iter));

        outImageNames.push_back(newFilePath.toStdString());
        outImageNumbers.push_back(dirNameIter);
    }
    write_csv(fn_csv, outImageNumbers, outImageNames);
}


void Dialog::on_removeImage_clicked()
{
    if(selimage)
    {
        imgSnapList.removeAt(selimage->pos);
        reloadImages();
        selimage = 0;
    }
}


/** @function detectAndDisplay */
void Dialog::detectAndDisplay( cv::Mat frame )
{
    gpuMatOriginal.upload(frame);

    cv::cuda::cvtColor( gpuMatOriginal, gpuMatProcessed, cv::COLOR_BGR2GRAY );
    cv::cuda::equalizeHist( gpuMatProcessed,  gpuMatProcessed2);

    cv::cuda::GpuMat objbuff;
    face_cascade_run->detectMultiScale( gpuMatProcessed2, objbuff );

    std::vector<Rect> faces;
    face_cascade_run->convert(objbuff, faces);

    for( size_t i = 0; i < faces.size(); i++ )
    {
      Point center( faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5 );
      ellipse( frame, center, Size( faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

      cv::cuda::GpuMat faceROI = gpuMatProcessed2( faces[i] );

      cv::cuda::GpuMat objbuff2;
      eyes_cascade_run->detectMultiScale( faceROI, objbuff2 );

      std::vector<Rect> eyes;
      eyes_cascade_run->convert(objbuff2, eyes);
      for( size_t j = 0; j < eyes.size(); j++ )
       {
         Point center( faces[i].x + eyes[j].x + eyes[j].width * 0.5, faces[i].y + eyes[j].y + eyes[j].height * 0.5 );
         int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
         circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
       }
    }

    cv::Mat frametemp;
    cvtColor(frame, frametemp, cv::COLOR_BGR2RGB);

    QImage qimgProcessed2((uchar*)frametemp.data, frametemp.cols, frametemp.rows, frametemp.step, QImage::Format_RGB888);
    ui->lblFeed->setPixmap(QPixmap::fromImage(qimgProcessed2));


}

void Dialog::on_trainButton_clicked()
{
    images.clear();
    labels.clear();

    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {

        ui->txtOutput->setText(QString::fromStdString("Error opening file \"" + fn_csv + "\". Reason: " + e.msg));
        //cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        return;
    }

    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        ui->txtOutput->setText(QString::fromStdString("This demo needs at least 2 images to work. Please add more images to your data set!"));
    }

    model->train(images, labels);
    model->save(modelfilepath.toStdString());
}
