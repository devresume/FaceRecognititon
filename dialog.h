#ifndef DIALOG_H
#define DIALOG_H

#include <sstream>
#include <string>
#include <QDialog>
#include <QSlider>
#include <QMessageBox>
#include <QString>
#include <QMenu>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include <QFileDialog>
#include <QGridLayout>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/face.hpp>
#include "imageitem.h"
#include <QHBoxLayout>
#include <QProgressBar>
#include "opencv2/cudaobjdetect.hpp"

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = 0);
    void reloadImages();
    void cannyAndDisplay(cv::Mat camMat);

    void clearLayout(QLayout* layout, bool deleteWidgets = true);
    ~Dialog();
    QList<cv::Mat> imgSnapList;
    imageItem* imgSel;
private slots:

    void on_btnImagePath_clicked();

    void on_btnCsvFile_clicked();

    void on_btnFindFaces_clicked();

    void processFrameAndUpdateGUI();

    void detectAndDisplay( cv::Mat frame );

    void on_btnSaveSnap_clicked();

    void on_btnExportFaces_clicked();

    //void ShowContextMenu(const QPoint&);

    void changeSelImage();

    void removeImage();


    void on_removeImage_clicked();

    void on_trainButton_clicked();

private:
    Ui::Dialog *ui;

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    std::string output_folder;
    std::string fn_csv;

    cv::VideoCapture capWebcam;
    cv::Mat matOriginal;
    cv::Mat matOriginal2;
    cv::Mat matOriginal3;

    cv::Mat resizeOriginal;
    cv::Mat matProcessed;

    cv::Mat originalSize2;
    cv::Mat originalSize3;

    QImage qimgProcessed;

    QTimer* tmrTimer;
    QGridLayout* imgGridLayout;
    QProgressBar* facerecProgress;

    int imgCount;

    QList<imageItem*> imgSnapWidgets;

    QMenu removeMenu;
    QAction* remAct;
    QHBoxLayout* imagePreviewLayout;
    int selImagePos;
    imageItem* selimage;
    QMenu* myMenu;

    QSizePolicy sizePolicyFixed;
    QSizePolicy sizePolicyExpand;


    QString face_cascade_name = "/home/devon/FaceRecognition/haarcascade_frontalface_alt.xml";
    QString eyes_cascade_name = "/home/devon/FaceRecognition/haarcascade_eye_tree_eyeglasses.xml";
    QString modelfilepath = "eigenfaces_atsave1.yml";

    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;
    cv::cuda::GpuMat gpuMatOriginal;
    cv::cuda::GpuMat gpuMatProcessed;
    cv::cuda::GpuMat gpuMatProcessed2;
    cv::Mat matProcessedOut;
//    cv::Ptr<cv::CascadeClassifier>  face_cascade;
//    cv::CascadeClassifier  eyes_cascade;
    cv::Ptr<cv::face::BasicFaceRecognizer> model;
    cv::Ptr<cv::cuda::CascadeClassifier> face_cascade_run;
    cv::Ptr<cv::cuda::CascadeClassifier> eyes_cascade_run;
};

#endif // DIALOG_H
