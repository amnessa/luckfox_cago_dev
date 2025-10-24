#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "yolov5.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define DISP_WIDTH  320
#define DISP_HEIGHT 240

// disp size
int width    = DISP_WIDTH;
int height   = DISP_HEIGHT;

// model size
int model_width = 640;
int model_height = 640;
float scale ;
int leftPadding ;
int topPadding  ;

static void *GetMediaBuffer(void *arg) {
    (void)arg;
    VENC_STREAM_S stFrame;
    stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
    while (1) {
        RK_S32 s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);
        if (s32Ret == RK_SUCCESS) {
            rtsp_demo_handle live = (rtsp_demo_handle)arg;
            extern rtsp_session_handle g_rtsp_session;
            if (live && g_rtsp_session) {
                void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
                rtsp_tx_video(g_rtsp_session, (uint8_t *)pData,
                              stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);
                rtsp_do_event(live);
            }
            RK_MPI_VENC_ReleaseStream(0, &stFrame);
        }
        usleep(10 * 1000);
    }
    free(stFrame.pstPack);
    return NULL;
}

cv::Mat letterbox(cv::Mat input)
{
    float scaleX = (float)model_width  / (float)width;
    float scaleY = (float)model_height / (float)height;
    scale = scaleX < scaleY ? scaleX : scaleY;

    int inputWidth   = (int)((float)width  * scale);
    int inputHeight  = (int)((float)height * scale);

    leftPadding = (model_width  - inputWidth) / 2;
    topPadding  = (model_height - inputHeight) / 2;

    cv::Mat inputScale;
    cv::resize(input, inputScale, cv::Size(inputWidth,inputHeight), 0, 0, cv::INTER_LINEAR);
    cv::Mat letterboxImage(model_height, model_width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect roi(leftPadding, topPadding, inputWidth, inputHeight);
    inputScale.copyTo(letterboxImage(roi));
    return letterboxImage;
}

void mapCoordinates(int *x, int *y) {
    int mx = *x - leftPadding;
    int my = *y - topPadding;
    *x = (int)((float)mx / scale);
    *y = (int)((float)my / scale);
}

rtsp_session_handle g_rtsp_session = NULL;

int main(int argc, char *argv[]) {
  system("RkLunch-stop.sh");
  // RKNN
  rknn_app_context_t rknn_app_ctx;
  object_detect_result_list od_results;
  memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
  const char *model_path = "./model/yolov5.rknn";
  if (init_yolov5_model(model_path, &rknn_app_ctx) != 0) {
      printf("init rknn failed\n");
      return -1;
  }
  printf("init rknn model success!\n");
  init_post_process();

  // rkaiq init
  RK_BOOL multi_sensor = RK_FALSE;
  const char *iq_dir = "/etc/iqfiles";
  rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
  SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
  SAMPLE_COMM_ISP_Run(0);

  // rkmpi init
  if (RK_MPI_SYS_Init() != RK_SUCCESS) {
      printf("rk mpi sys init fail!\n");
      return -1;
  }

  // rtsp init (use 8554 to avoid perms)
  rtsp_demo_handle g_rtsplive = create_rtsp_demo(8554);
  g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
  rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
  rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

  // vi init: ch0 for encoder, ch1 for AI
  vi_dev_init();
  vi_chn_init(0, width, height);
  vi_chn_init(1, width, height);

  // venc init (NV12)
  RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
  venc_init(0, width, height, enCodecType);

  // bind vi0 -> venc0
  MPP_CHN_S stSrcChn, stDestChn;
  stSrcChn.enModId = RK_ID_VI;    stSrcChn.s32DevId = 0; stSrcChn.s32ChnId = 0;
  stDestChn.enModId = RK_ID_VENC; stDestChn.s32DevId = 0; stDestChn.s32ChnId = 0;
  if (RK_MPI_SYS_Bind(&stSrcChn, &stDestChn) != RK_SUCCESS) {
      printf("bind vi0->venc0 failed\n");
      return -1;
  }
  printf("stream path bound (VI0->VENC0)\n");

  // start stream thread
  pthread_t stream_th;
  pthread_create(&stream_th, NULL, GetMediaBuffer, (void*)g_rtsplive);

  // detection loop on VI ch1
  VIDEO_FRAME_INFO_S stViFrame;
  cv::Mat model_bgr(model_height, model_width, CV_8UC3);
  while (1) {
      RK_S32 s32Ret = RK_MPI_VI_GetChnFrame(0, 1, &stViFrame, -1);
      if (s32Ret != RK_SUCCESS) continue;

      void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);
      cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, vi_data);
      cv::Mat bgr(height, width, CV_8UC3);
      cv::cvtColor(yuv420sp, bgr, cv::COLOR_YUV420sp2BGR);

      // preprocess
      cv::Mat letter = letterbox(bgr);
      memcpy(rknn_app_ctx.input_mems[0]->virt_addr, letter.data, model_width * model_height * 3);
      inference_yolov5_model(&rknn_app_ctx, &od_results);

      // TODO: use RGN overlay here if you want boxes on stream (optional)

      RK_MPI_VI_ReleaseChnFrame(0, 1, &stViFrame);
      // small sleep to reduce CPU
      usleep(1000);
  }

  // cleanup (unreached in this loop)
  pthread_join(stream_th, NULL);
  RK_MPI_SYS_UnBind(&stSrcChn, &stDestChn);
  RK_MPI_VI_DisableChn(0, 0);
  RK_MPI_VI_DisableChn(0, 1);
  RK_MPI_VENC_StopRecvFrame(0);
  RK_MPI_VENC_DestroyChn(0);
  RK_MPI_VI_DisableDev(0);
  if (g_rtsplive) rtsp_del_demo(g_rtsplive);
  RK_MPI_SYS_Exit();
  release_yolov5_model(&rknn_app_ctx);
  deinit_post_process();
  return 0;
}
//TODO: add fps metric, choose single tag to follow and draw distance from center.