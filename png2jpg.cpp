#include <iostream>
#include <opencv2/opencv.hpp>
#include <io.h>

#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>

using namespace std;
using namespace cv;

void getFiles(string path, vector<string> &files)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir = opendir(path.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }
    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8) ///file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            files.push_back(ptr->d_name);
        else if (ptr->d_type == 10) ///link file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            continue;
        else if (ptr->d_type == 4) ///dir
        {
            files.push_back(ptr->d_name);
            /* 
                memset(base,'\0',sizeof(base)); 
                strcpy(base,basePath); 
                strcat(base,"/"); 
                strcat(base,ptr->d_nSame); 
                readFileList(base); 
            */
        }
    }
    closedir(dir);

    for (int i = 0; i < files.size(); i++)
    {
        cout << "files[" << i << "] = " << files[i] << endl;
    }
}

int main()
{
    string path = "/home/wys/slam/data/zhuhai/0903/right/";
    vector<string> files;
    getFiles(path, files);

    cv::Mat image;
    cout << "files.size() = " << files.size() << endl;
    for (int i = 0; i < files.size(); i++)
    {
        image = imread(path + files[i], 1);
        string str = files[i].substr(0, files[i].length() - 4);
        string pngname = path + str + ".jpg";
        cout << pngname << endl;
        imwrite(pngname,image);
    }
}
