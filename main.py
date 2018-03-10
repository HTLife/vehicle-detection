from moviepy.editor import VideoFileClip
from svm_pipeline import *
from yolo_pipeline import *
from lane import *
#import matplotlib.image as mpimg

def pipeline_yolo(img):

    # img_undist, img_lane_augmented, lane_info = lane_process(img)
    # output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)
    output = vehicle_detection_yolo_(img)

    return output

def pipeline_svm(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_svm(img_undist, img_lane_augmented, lane_info)

    return output


def proc_img(f_path, filename):
        
    image = mpimg.imread(f_path)
    # try:
    yolo_result = pipeline_yolo(image)

    cv2.imwrite('/media/rvl/hddData1/ITRI/dataset/front0_out/' + filename, cv2.cvtColor(yolo_result, cv2.COLOR_RGB2BGR))
        #mpimg.close()
    # except:
        # print('exception happened...')
        #mpimg.close()

if __name__ == "__main__":

    demo = 1  # 1:image (YOLO and SVM), 2: video (YOLO Pipeline), 3: video (SVM pipeline)

    if demo == 1:
        #filename = '/media/rvl/hddData1/ITRI/dataset/jpg/frame0113.jpg'

        for filename in os.listdir('/media/rvl/hddData1/ITRI/dataset/front0_jpg/'):
            f_path = '/media/rvl/hddData1/ITRI/dataset/front0_jpg/' + filename
            proc_img(f_path, filename)
            
            #(1) Yolo pipeline
            
        # filen = 'frame0498.jpg'
        # filename = '/media/rvl/hddData1/ITRI/dataset/front0_jpg/' + filen
        # image = mpimg.imread(filename)

        # #(1) Yolo pipeline
        # yolo_result = pipeline_yolo(image)
        # cv2.imwrite('/media/rvl/hddData1/ITRI/dataset/front0_out/' + filen, cv2.cvtColor(yolo_result, cv2.COLOR_RGB2BGR))
        

        # plt.figure()
        # plt.imshow(yolo_result)
        # plt.title('yolo pipeline', fontsize=30)

        #(2) SVM pipeline
        # draw_img = pipeline_svm(image)
        # fig = plt.figure()
        # plt.imshow(draw_img)
        # plt.title('svm pipeline', fontsize=30)
        # plt.show()

    elif demo == 2:
        # YOLO Pipeline
        video_output = 'examples/project_YOLO.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4").subclip(30,32)
        clip = clip1.fl_image(pipeline_yolo)
        clip.write_videofile(video_output, audio=False)

    else:
        # SVM pipeline
        video_output = 'examples/project_svm.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4").subclip(30,32)
        clip = clip1.fl_image(pipeline_svm)
        clip.write_videofile(video_output, audio=False)


