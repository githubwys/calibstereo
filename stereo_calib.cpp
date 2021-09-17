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
                imshow("corners" + filename, cimg1);
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

    double rms1, rms2;
    vector<Mat> rvecs, tvecs;
    int flags = 0;

    std::cout << "========initCameraMatrix2D========" << std::endl;
    cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    cout << "cameraMatrix[0] = " << cameraMatrix[0] << endl;
    //cout << "distCoeffs[0] = " << distCoeffs[0] << endl;
    cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
    cout << "cameraMatrix[1] = " << cameraMatrix[1] << endl;
    //cout << "distCoeffs[1] = " << distCoeffs[1] << endl;
    cv::Mat stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;
    // rms1 = calibrateCamera(objectPoints, imagePoints[0], imageSize,      //| CV_CALIB_FIX_K3 | CALIB_USE_LU
    //                        cameraMatrix[0], distCoeffs[0], rvecs, tvecs, //distCoeffs:Output vector of distortion coefficients
    //                        CV_CALIB_FIX_K3, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));
    // rms2 = calibrateCamera(objectPoints, imagePoints[1], imageSize,
    //                        cameraMatrix[1], distCoeffs[1], rvecs, tvecs, //distCoeffs:Output vector of distortion coefficients
    //                        CV_CALIB_FIX_K3, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));
    rms1 = calibrateCamera(objectPoints, imagePoints[0], imageSize,      //| CV_CALIB_FIX_K3 | CALIB_USE_LU
                           cameraMatrix[0], distCoeffs[0], rvecs, tvecs, //distCoeffs:Output vector of distortion coefficients
                           stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors,
                           0, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 40, DBL_EPSILON));
    rms2 = calibrateCamera(objectPoints, imagePoints[1], imageSize,      //| CV_CALIB_FIX_K3 | CALIB_USE_LU
                           cameraMatrix[1], distCoeffs[1], rvecs, tvecs, //distCoeffs:Output vector of distortion coefficients
                           stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors,
                           0, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 40, DBL_EPSILON));
    std::cout << "========calibrateCamera========" << std::endl;
    cout << "cameraMatrix[0] = " << cameraMatrix[0] << endl;
    cout << "distCoeffs[0] = " << distCoeffs[0] << endl;
    cout << "cameraMatrix[1] = " << cameraMatrix[1] << endl;
    cout << "distCoeffs[1] = " << distCoeffs[1] << endl;

    // // cv::Mat K1 = cv::Mat(cv::Matx33d(
    // //     1643.23922244931, 0, 955.632389024248,
    // //     0, 1646.60777233688, 491.359781135353,
    // //     0, 0, 1));
    // cv::Mat K1 = cv::Mat(cv::Matx33d(
    //     1.9937156242710637e+03, 0., 8.8059499438253215e+02, 0.,
    //     1.9978872214999717e+03, 5.5269502230818307e+02, 0., 0., 1.));
    // // cv::Mat distCoeffs1 = cv::Mat(cv::Matx41d(
    // //     -0.409946054902810, 0.179241711483436,
    // //     -0.000713866588082986, 0.000617293005958004));
    // // cv::Mat distCoeffs1 = cv::Mat(cv::Matx41d(
    // //      -2.4352270240974530e-01, 1.8469288881955251e-01,
    // //    -1.8951658401333046e-03, -8.6162043491556115e-04));
    // cv::Mat distCoeffs1 = (cv::Mat_<double>(5, 1) << -2.4352270240974530e-01, 1.8469288881955251e-01,
    //                        -1.8951658401333046e-03, -8.6162043491556115e-04, 0.);
    // // cv::Mat K2 = cv::Mat(cv::Matx33d(
    // //     1994.72443215177, 0, 881.901669489577,
    // //     0, 1998.44496010574, 556.330703994452,
    // //     0, 0, 1));
    // cv::Mat K2 = cv::Mat(cv::Matx33d(
    //     1.6515124810046834e+03, 0., 9.5641812787368167e+02, 0.,
    //     1.6532627919168356e+03, 5.0407186507061311e+02, 0., 0., 1.));
    // // cv::Mat distCoeffs2 = cv::Mat(cv::Matx41d(
    // //     -0.243547179711356, 0.183777511637158,
    // //     -0.00165050761251720, -0.000625729681232135));
    // // cv::Mat distCoeffs2 = cv::Mat(cv::Matx41d(
    // //     -4.0711389870291248e-01, 1.7521138732705704e-01,
    // //     -1.5060062330082536e-03, 3.8545430851559504e-04));
    // cv::Mat distCoeffs2 = (cv::Mat_<double>(5, 1) << -4.0711389870291248e-01, 1.7521138732705704e-01,
    //                        -1.5060062330082536e-03, 3.8545430851559504e-04, 0.);
    // // cameraMatrix[0] = K1;
    // // cameraMatrix[1] = K2;
    // // distCoeffs[0] = distCoeffs1;
    // // distCoeffs[1] = distCoeffs2;
    //cout << "objectPoints.at(1) = " << objectPoints.at(1) << endl;
    //cout << "cameraMatrix[0] = " << cameraMatrix[0] << endl;
    //cout << "cameraMatrix[1] = " << cameraMatrix[1] << endl;
    Mat R, T, E, F; //R旋转矩阵 T平移矩阵 E本征矩阵 F输出基本矩阵

    cv::Mat K1 = (cv::Mat_<double>(3, 3) << 1210.80848799380, 0, 1030.15171709705,
                  0, 1200.72633122232, 526.926907710212,
                  0, 0, 1);
    cv::Mat K2 = (cv::Mat_<double>(3, 3) << 1148.29076225360, 0, 933.844539548117,
                  0, 1148.04008128052, 549.669448597396,
                  0, 0, 1);
    cv::Mat diff1 = (cv::Mat_<double>(4, 1) << -0.340457891725979, 0.125829308691339,
                     -0.00179305443517940, -0.0182089027925910);
    cv::Mat diff2 = (cv::Mat_<double>(4, 1) << -0.363061254145000, 0.125601126142260,
                     -0.000695062925281247, 0.00904383492966770);
    cameraMatrix[0] = K1;
    cameraMatrix[1] = K2;
    distCoeffs[0] = diff1;
    distCoeffs[1] = diff2;
    //undistort check
    //cv::undistort();
    for (i = j = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++) //依次寻找左右图片
        {
            const string &filename = imagelist[i * 2 + k];
            Mat img = imread(filename, 1); //载入灰度图 0代表灰度图
            cv::Mat undistortimage;
            cv::undistort(img, undistortimage, cameraMatrix[k], distCoeffs[k]);
            cv::resize(undistortimage, undistortimage, cv::Size(undistortimage.cols / 2, undistortimage.rows / 2), 0, 0, CV_INTER_LINEAR);
            string str;
            if (k == 0)
            {
                str = "left" + filename;
            }
            else
            {
                str = "right " + filename;
            }
            cv::imshow(str, undistortimage);
            cv::waitKey(1);
        }
    }

    //最关键的地方，求解校正后的相机参数
    //CALIB_FIX_INTRINSIC
    // double rms = stereoCalibrate(objectPoints, imagePoints[1], imagePoints[0],
    //                              cameraMatrix[1], distCoeffs[1],
    //                              cameraMatrix[0], distCoeffs[0],
    //                              imageSize, R, T, E, F,
    //                              CV_CALIB_FIX_INTRINSIC,
    //                              TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 9.999999999999999547e-07));
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],//迭代太多有问题啊啊啊啊
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 CV_CALIB_FIX_INTRINSIC,
                                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1, 100));//以matlab为准。。。
    // double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
    //                              cameraMatrix[0], distCoeffs[0],
    //                              cameraMatrix[1], distCoeffs[1],
    //                              imageSize, R, T, E, F,
    //                              CALIB_FIX_ASPECT_RATIO +
    //                                  CALIB_ZERO_TANGENT_DIST +
    //                                  CALIB_USE_INTRINSIC_GUESS +
    //                                  CALIB_SAME_FOCAL_LENGTH +
    //                                  CALIB_RATIONAL_MODEL +
    //                                  CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5+ CALIB_FIX_K6,
    //                              TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
    cout << "done with RMS error=" << rms << endl;
    cout << "cameraMatrix[0] = " << cameraMatrix[0] << endl;
    cout << "cameraMatrix[1] = " << cameraMatrix[1] << endl;
    cout << "distCoeffs[0] = " << distCoeffs[0] << endl;
    cout << "distCoeffs[1] = " << distCoeffs[1] << endl;
    cout << "R = " << R << endl;
    cout << "T = " << T << endl;
    cout << "E = " << E << endl;
    cout << "F = " << F << endl;
    // F = (cv::Mat_<double>(3, 3) << -2.90365098525388e-06, -2.43985429687037e-05, -0.0135282464577077,
    //      2.22113720288687e-05, -1.39978509403813e-06, -0.130553767933778,
    //      0.0361089749372620, 0.150398759722962, -22.2154654202392);
    F = (cv::Mat_<double>(3, 3) << -1.29240022127073e-06, -1.42218285664285e-05, 0.0357494515015725,
         -9.07755103850985e-05, -4.16352346426982e-05, 1.51946364371862,
         0.0995592761299502, -1.30974600063752, -262.620091744838);
    F = F.t();
    cout << "F = " << F << endl;
    F = F * (1.0 / -262.620091744838);
    cout << "F = " << F << endl;
    R = (cv::Mat_<double>(3, 3) << 0.994438182887672, -0.0557182425785059, -0.0893766068892123,
         0.0526727439156626, 0.997959798194518, -0.0360807876293297,
         0.0912046186520236, 0.0311724017605403, 0.995344161034272);
    R = R.t();
    T = (cv::Mat_<double>(3, 1) << -1617.70652117081, 31.6409906914789, 18.5049367438881);
    E = (cv::Mat_<double>(3, 3) << -1.79690183222168, -19.6088148040537, 30.9168320113695,
         -126.183104134755, -57.3934196426593, 1611.86247581333,
         58.6707550794259, -1616.07269120550, -53.3136020983709);
    E = E.t();

    // //undistort check
    // //cv::undistort();
    // for (i = j = 0; i < nimages; i++)
    // {
    //     for (k = 0; k < 2; k++) //依次寻找左右图片
    //     {
    //         const string &filename = imagelist[i * 2 + k];
    //         Mat img = imread(filename, 0); //载入灰度图 0代表灰度图
    //         cv::Mat undistortimage;
    //         cv::undistort(img, undistortimage, cameraMatrix[k], distCoeffs[k]);
    //         cv::resize(undistortimage, undistortimage, cv::Size(undistortimage.cols / 2, undistortimage.rows / 2), 0, 0, CV_INTER_LINEAR);
    //         cv::imshow(filename, undistortimage);
    //         cv::waitKey(0);
    //     }
    // }

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
        Mat imgptundist[2];
        //cv::Mat img[2];
        for (k = 0; k < 2; k++)
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgptundist[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            // string filename = imagelist[i * 2 + k]; //left 0   ;right  1
            // if (k == 0)
            // {
            //     cv::Mat img = imread(filename, 0); //载入灰度图 0代表灰度图
            //     cv::Mat imgundist;
            //     //cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2), 0, 0, INTER_LINEAR);
            //     //cv::imshow("distort" + filename, img);
            //     //cv::waitKey(500);
            //     cv::undistort(img, imgundist, cameraMatrix[k], distCoeffs[k]);
            //     cv::resize(imgundist, imgundist, cv::Size(imgundist.cols / 2, imgundist.rows / 2), 0, 0, INTER_LINEAR);
            //     cv::imshow("undistort" + filename, imgundist);
            //     cv::waitKey(1);
            // }
            computeCorrespondEpilines(imgptundist[k], k + 1, F, lines[k]);
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
    cv::CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|0.020|}{nr||}{help||}{@input|/home/wys/slam/camera-calib/calibstereo/stereo_calib2.xml|}");
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

    StereoCalib(imagelist, boardSize, squareSize, false, true, showRectified);
    return 0;
}
