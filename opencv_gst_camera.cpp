#include "b.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <ncurses.h>
#include <unistd.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
///gpio
#include <termios.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include "jetsonGPIO.h" //a C header, so wrap it in extern "C" 
//V4L2
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

 
#define u8 unsigned char
#define  LOGD(...)  do {printf(__VA_ARGS__);printf("\n");} while (0)
#define DBG(fmt, args...) LOGD("%s:%d, " fmt, __FUNCTION__, __LINE__, ##args);
#define ASSERT(b) \
do \
{ \
    if (!(b)) \
    { \
        LOGD("error on %s:%d", __FUNCTION__, __LINE__); \
        return 0; \
    } \
} while (0)
 
#define VIDEO_DEVICE "/dev/video0"
#define IMAGE_WIDTH 2592//sensor resolution = 2592*1458
#define IMAGE_HEIGHT 1458
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * 2)
#define BUFFER_COUNT 5//申请5個buffer

extern void CalcImageConvolution_cuda(unsigned short *data,long w,long h);

using namespace cv;
using namespace std;

 
int cam_fd = -1;
struct v4l2_buffer video_buffer[BUFFER_COUNT];
u8* video_buffer_ptr[BUFFER_COUNT];
u8 buf[IMAGE_SIZE];
 
int cam_open()
{
    cam_fd = open(VIDEO_DEVICE, O_RDWR);//connect camera
 
    if (cam_fd >= 0) return 0;
    else return -1;
}
 
int cam_close()
{
    close(cam_fd);//disconnect camera
 
    return 0;
}
 
int cam_select(int index)
{
    int ret;
 
    int input = index;
    ret = ioctl(cam_fd, VIDIOC_S_INPUT, &input);//setting video input
    return ret;
}
 
int cam_init()
{
    int i;
    int ret;
    struct v4l2_format format;
 
    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;//frame type,用於視頻捕獲設備
    //format.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR8;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR10;//V4L2_PIX_FMT_SGRBG10;//10bit raw
    format.fmt.pix.width = IMAGE_WIDTH;//resolution
    format.fmt.pix.height = IMAGE_HEIGHT;
    ret = ioctl(cam_fd, VIDIOC_TRY_FMT, &format);//設置當前格式
    if (ret != 0)
    {
        DBG("ioctl(VIDIOC_TRY_FMT) failed %d(%s)", errno, strerror(errno));
        return ret;
    }
 
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = ioctl(cam_fd, VIDIOC_S_FMT, &format);//設置當前格式
    if (ret != 0)
    {
        DBG("ioctl(VIDIOC_S_FMT) failed %d(%s)", errno, strerror(errno));
        return ret;
    }
 
    struct v4l2_requestbuffers req;
    req.count = BUFFER_COUNT;//buffer個數
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;//buffer type
    req.memory = V4L2_MEMORY_MMAP;//內存映射方式
    ret = ioctl(cam_fd, VIDIOC_REQBUFS, &req);//申請buffer
    if (ret != 0)
    {
        DBG("ioctl(VIDIOC_REQBUFS) failed %d(%s)", errno, strerror(errno));
        return ret;
    }
    DBG("req.count: %d", req.count);
    if (req.count < BUFFER_COUNT)
    {
        DBG("request buffer failed");
        return ret;
    }
 
    struct v4l2_buffer buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.type = req.type;
    buffer.memory = V4L2_MEMORY_MMAP;
    for (i=0; i<req.count; i++)
    {
        buffer.index = i;
        ret = ioctl (cam_fd, VIDIOC_QUERYBUF, &buffer);//獲取buffer address
        if (ret != 0)
        {
            DBG("ioctl(VIDIOC_QUERYBUF) failed %d(%s)", errno, strerror(errno));
            return ret;
        }
        DBG("buffer.length: %d", buffer.length);
        DBG("buffer.m.offset: %d", buffer.m.offset);
        video_buffer_ptr[i] = (u8*) mmap(NULL, buffer.length, PROT_READ|PROT_WRITE, MAP_SHARED, cam_fd, buffer.m.offset);//内存映射
        if (video_buffer_ptr[i] == MAP_FAILED)
        {
            DBG("mmap() failed %d(%s)", errno, strerror(errno));
            return -1;
        }
 
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;
        ret = ioctl(cam_fd, VIDIOC_QBUF, &buffer);//把buffer放入queue中
        if (ret != 0)
        {
            DBG("ioctl(VIDIOC_QBUF) failed %d(%s)", errno, strerror(errno));
            return ret;
        }
    }
 
    int buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = ioctl(cam_fd, VIDIOC_STREAMON, &buffer_type);//啟動數據流
    if (ret != 0)
    {
        DBG("ioctl(VIDIOC_STREAMON) failed %d(%s)", errno, strerror(errno));
        return ret;
    }
 
    DBG("cam init done.");
 
    return 0;
}
 
int cam_get_image(u8* out_buffer, int out_buffer_size)
{
    int ret;
    struct v4l2_buffer buffer;
 
    memset(&buffer, 0, sizeof(buffer));
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = BUFFER_COUNT;
    ret = ioctl(cam_fd, VIDIOC_DQBUF, &buffer);//從queue中取出一幀
    if (ret != 0)
    {
        DBG("ioctl(VIDIOC_DQBUF) failed %d(%s)", errno, strerror(errno));
        return ret;
    }
 
    if (buffer.index < 0 || buffer.index >= BUFFER_COUNT)
    {
        DBG("invalid buffer index: %d", buffer.index);
        return ret;
    }
 
    DBG("dequeue done, index: %d", buffer.index);
    memcpy(out_buffer, video_buffer_ptr[buffer.index], IMAGE_SIZE);//拷貝buffer data出來
    DBG("copy done.");
 
    ret = ioctl(cam_fd, VIDIOC_QBUF, &buffer);//buffer放入queue中
    if (ret != 0)
    {
        DBG("ioctl(VIDIOC_QBUF) failed %d(%s)", errno, strerror(errno));
        return ret;
    }
    DBG("enqueue done.");
 
    return 0;
}

int gst_pipeline()
{
    int i;
    int ret;
 
    ret = cam_open();
    ASSERT(ret==0);
 
    ret = cam_select(0);
    ASSERT(ret==0);
 
    ret = cam_init();
    ASSERT(ret==0);
 
    int count = 0;
    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);

    scrollok(stdscr, TRUE);

    while (true)
    {
        ret = cam_get_image(buf, IMAGE_SIZE);
        ASSERT(ret==0);
 
        char tmp[64] = {"---\n"};
        for (i=0; i<16; i++)
            sprintf(&tmp[strlen(tmp)], "%02x ", buf[i]);
        LOGD("%s", tmp);
	CalcImageConvolution_cuda((unsigned short*)buf,IMAGE_WIDTH,IMAGE_HEIGHT);

	// 'q' for termination
	if (getch() == 'q' )
	{
	    endwin();
	    break;
	}

    }

 
    ret = cam_close();
    ASSERT(ret==0);
 
    return 0;
}
