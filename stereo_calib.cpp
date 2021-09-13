/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OPENCV WEBSITES:
     Homepage:      http://opencv.org
     Online docs:   http://docs.opencv.org
     Q&A forum:     http://answers.opencv.org
     GitHub:        https://github.com/opencv/opencv/
   ************************************************** */

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

#define INTER_LINEAR_EXACT 5

static int print_help()
{
    cout << " Given a list of chessboard images, the number of corners (nx, ny)\n"
            " on the chessboards, and a flag: useCalibrated for \n"
            "   calibrated (0) or\n"
            "   uncalibrated \n"
            "     (1: use stereoCalibrate(), 2: compute fundamental\n"
            "         matrix separately) stereo. \n"
            " Calibrate the cameras and display the\n"
            " rectified results along with the computed disparity images.   \n"
         << endl;
    cout << "Usage:\n ./stereo_calib -w=<board_width default=9> -h=<board_height default=6> -s=<square_size default=1.0> <image list XML/YML file default=stereo_calib.xml>\n"
         << endl;
    return 0;
}

static void
StereoCalib(const vector<string> &imagelist, Size boardSize, float squareSize, bool displayCorners = false, bool useCalibrated = true, bool showRectified = true)
{
    if (imagelist.size() % 2 != 0) //判断图片是否成对
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 2; //设定寻找角点的图像尺寸，若scale未找到，则将图像放大寻找角点
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f>> imagePoints[2];
    vector<vector<Point3f>> objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size() / 2; //左右图像的个数

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

    for (i = j = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++) //依次寻找左右图片
        {
            const string &filename = imagelist[i * 2 + k];
            Mat img = imread(filename, 0); //载入灰度图 0代表灰度图
            cout << filename << endl;
            if (img.empty())
                break;
            if (imageSize == Size()) //判断图像尺寸是否达到预先设置的要求,k=0:（左图）第一张图赋值
                imageSize = img.size();
            else if (img.size() != imageSize) //k=1:（右图）第二张图赋值，size相等
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            //设置图像矩阵的引用(指针)，此时指向左右视图的矩阵首地址
            vector<Point2f> &corners = imagePoints[k][j]; //赋值角点个数及位置
            for (int scale = 1; scale <= maxScale; scale++)
            {
                Mat timg;
                if (scale == 1)
                    timg = img;
                else
                    // resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
                    resize(img, timg, Size(), scale, scale, 1);         //opencv version problem
                found = findChessboardCorners(timg, boardSize, corners, //找角点函数，得到内角点（像素）坐标
                                              CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if (found)
                {
                    if (scale > 1)
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1. / scale;
                    }
                    break;
                }
            }
            if (displayCorners)
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                // cv::imshow("img", img);
                // cv::waitKey(0);
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found); //画出角点，自己调试时使用，但会降低精度??，正式运行时应该注释掉
                double sf = 640. / MAX(img.rows, img.cols);
                //cv::imshow("img", img);
                //cv::waitKey(0);
                //resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);//opencv version problem
                resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR); //
                imshow("corners", cimg1);
                //imshow("corners", cimg);
                char c = (char)waitKey(500);
                if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if (!found)
                break;
            cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
                                      30, 0.01));
        }
        if (k == 2)
        {
            goodImageList.push_back(imagelist[i * 2]);
            goodImageList.push_back(imagelist[i * 2 + 1]);
            std::cout << "success pair = " << imagelist[i * 2] << " & " << imagelist[i * 2 + 1] << std::endl;
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j; //nimages为左右图像个数，success
    if (nimages < 2)
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages); //左相机 角点位置
    imagePoints[1].resize(nimages); //右相机 角点位置
    objectPoints.resize(nimages);

    for (i = 0; i < nimages; i++)
    {
        for (j = 0; j < boardSize.height; j++)
            for (k = 0; k < boardSize.width; k++)
                objectPoints[i].push_back(Point3f(k * squareSize, j * squareSize, 0)); //棋盘格物方点
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2]; //相机参数  畸变矩阵
    cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
    cv::Mat K1 = cv::Mat(cv::Matx33d(
        1643.23922244931, 0, 955.632389024248,
        0, 1646.60777233688, 491.359781135353,
        0, 0, 1));
    cv::Mat distCoeffs1 = cv::Mat(cv::Matx41d(
        -0.409946054902810, 0.179241711483436,
        -0.000713866588082986, 0.000617293005958004));
    cv::Mat K2 = cv::Mat(cv::Matx33d(
        1994.72443215177, 0, 881.901669489577,
        0, 1998.44496010574, 556.330703994452,
        0, 0, 1));
    cv::Mat distCoeffs2 = cv::Mat(cv::Matx41d(
        -0.243547179711356, 0.183777511637158,
        -0.00165050761251720, -0.000625729681232135));
    // cameraMatrix[0] = K1;
    // cameraMatrix[1] = K2;
    // distCoeffs[0] = distCoeffs1;
    // distCoeffs[1] = distCoeffs2;
    //cout << "objectPoints.at(1) = " << objectPoints.at(1) << endl;
    cout << "cameraMatrix[0] = " << cameraMatrix[0] << endl;
    cout << "cameraMatrix[1] = " << cameraMatrix[1] << endl;
    Mat R, T, E, F; //R旋转矩阵 T平移矩阵 E本征矩阵 F输出基本矩阵

    //undistort
    //cv::undistort();
    for (i = j = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++) //依次寻找左右图片
        {
            const string &filename = imagelist[i * 2 + k];
            Mat img = imread(filename, 0); //载入灰度图 0代表灰度图
            cv::Mat undistortimage;
            cv::undistort(img,undistortimage,cameraMatrix[k],distCoeffs[k]);
            cv::imshow("undistortimage",undistortimage);
            cv::waitKey(0);
        }
    }

    //最关键的地方，求解校正后的相机参数
    //CALIB_FIX_INTRINSIC
    // double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
    //                              cameraMatrix[0], distCoeffs[0],
    //                              cameraMatrix[1], distCoeffs[1],
    //                              imageSize, R, T, E, F,
    //                              CALIB_FIX_INTRINSIC,
    //                              TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 CALIB_FIX_ASPECT_RATIO +
                                     CALIB_ZERO_TANGENT_DIST +
                                     CALIB_USE_INTRINSIC_GUESS +
                                     CALIB_SAME_FOCAL_LENGTH +
                                     CALIB_RATIONAL_MODEL +
                                     CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
    cout << "done with RMS error=" << rms << endl;
    cout << "cameraMatrix[0] = " << cameraMatrix[0] << endl;
    cout << "cameraMatrix[1] = " << cameraMatrix[1] << endl;
    cout << "distCoeffs[0] = " << distCoeffs[0] << endl;
    cout << "distCoeffs[1] = " << distCoeffs[1] << endl;

    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0
    double err = 0; //计算投影标定误差
    int npoints = 0;
    vector<Vec3f> lines[2];       //极线
    for (i = 0; i < nimages; i++) //水平校正
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for (k = 0; k < 2; k++)
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
        }
        for (j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x * lines[1][j][0] +
                                imagePoints[0][i][j].y * lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x * lines[0][j][0] +
                                imagePoints[1][i][j].y * lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " << err / npoints << endl;

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] << "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

    // COMPUTE AND DISPLAY RECTIFICATION
    if (!showRectified)
        return;

    Mat rmap[2][2];
    // IF BY CALIBRATED (BOUGUET'S METHOD)
    if (useCalibrated)
    {
        // we already computed everything
    }
    // OR ELSE HARTLEY'S METHOD
    else
    // use intrinsic parameters of each camera, but
    // compute the rectification transformation directly
    // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for (k = 0; k < 2; k++)
        {
            for (i = 0; i < nimages; i++)
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv() * H1 * cameraMatrix[0];
        R2 = cameraMatrix[1].inv() * H2 * cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    Mat canvas_orig;
    double sf;
    int w, h;
    if (!isVerticalStereo)
    {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h, w * 2, CV_8UC3);
        canvas_orig.create(h, w * 2, CV_8UC3);
    }
    else
    {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h * 2, w, CV_8UC3);
        canvas_orig.create(h * 2, w, CV_8UC3);
    }

    for (i = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++)
        {
            Mat img = imread(goodImageList[i * 2 + k], 0), rimg, cimg, gimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            cvtColor(img, gimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w * k, 0, w, h)) : canvas(Rect(0, h * k, w, h));
            Mat canvasPart_orig = !isVerticalStereo ? canvas_orig(Rect(w * k, 0, w, h)) : canvas_orig(Rect(0, h * k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            resize(gimg, canvasPart_orig, canvasPart_orig.size(), 0, 0, INTER_AREA);
            if (useCalibrated)
            {
                Rect vroi(cvRound(validRoi[k].x * sf), cvRound(validRoi[k].y * sf),
                          cvRound(validRoi[k].width * sf), cvRound(validRoi[k].height * sf));
                rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
                rectangle(canvasPart_orig, vroi, Scalar(0, 0, 255), 3, 8);
            }
        }

        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
            {
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
                line(canvas_orig, Point(0, j), Point(canvas_orig.cols, j), Scalar(0, 255, 0), 1, 8);
            }
        else
            for (j = 0; j < canvas.cols; j += 16)
            {
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
                line(canvas_orig, Point(j, 0), Point(j, canvas_orig.rows), Scalar(0, 255, 0), 1, 8);
            }
        imshow("rectified", canvas);
        imshow("unrectified", canvas_orig);
        char c = (char)waitKey();
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }
}

static bool readStringList(const string &filename, vector<string> &l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
    {
        l.push_back((string)*it);
        //std::cout << "image = " << (string)*it << std::endl;
    }
    return true;
}

int main(int argc, char **argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified;
    cv::CommandLineParser parser(argc, argv, "{w|9|}{h|7|}{s|0.020|}{nr||}{help||}{@input|/home/wys/slam/camera-co-calib/calibStereo/stereo_calib.xml|}");
    if (parser.has("help"))
        return print_help();
    showRectified = !parser.has("nr");
    //imagelistfn = samples::findFile(parser.get<string>("@input"));
    std::cout << "parser.get<string>(input) = " << parser.get<string>("@input") << std::endl;
    imagelistfn = parser.get<string>("@input");
    std::cout << "imagelistfn = " << imagelistfn << std::endl;
    boardSize.width = parser.get<int>("w");
    std::cout << "boardSize.width = " << boardSize.width << std::endl;
    boardSize.height = parser.get<int>("h");
    std::cout << "boardSize.height = " << boardSize.height << std::endl;
    float squareSize = parser.get<float>("s");
    std::cout << "squareSize = " << squareSize << std::endl;
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if (!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return print_help();
    }

    for (auto it = imagelist.begin(); it < imagelist.end(); it++)
    {
        std::cout << "image.name = " << (string)*it << std::endl;
    }

    StereoCalib(imagelist, boardSize, squareSize, true, true, showRectified);
    return 0;
}
