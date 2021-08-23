import numpy as np 
import os
import configparser
import cv2

from utils import FileReader
from utils import FileWriter
from utils import KalmanFilter

def format_img_filename(frame_id, extension):
    str_id = str(frame_id)
    while len(str_id) < 6:
        str_id = "0"+str_id
    return str_id + extension

def IoU(bb1, bb2):
    if bb1[2]*bb1[3] == 0.0 or bb2[2]*bb2[3] == 0.0:
        return 0.0

    x_inter1 = np.max([bb1[1], bb2[1]])
    y_inter1 = np.max([bb1[0], bb2[0]])

    x_inter2 = np.min([bb1[1]+bb1[3], bb2[1]+bb2[3]])
    y_inter2 = np.min([bb1[0]+bb1[2], bb2[0]+bb2[2]])

    w_inter = (x_inter2-x_inter1)
    h_inter = (y_inter2-y_inter1)
    area_inter = w_inter*h_inter

    area_union = bb1[2]*bb1[3]+bb2[2]*bb2[3]-area_inter
    return area_inter/area_union

def main():
    DATASET_FOLDER = os.path.join(os.getcwd(), 'MOT15')
    TRAIN_FOLDER = os.path.join(DATASET_FOLDER, 'train')
    # TEST_FOLDER = os.path.join(DATASET_FOLDER, 'test')

    # Output folder
    OUTPUT_FOLDER = os.path.join(os.getcwd(), 'out')
    if os.path.exists(OUTPUT_FOLDER) == False:
        os.mkdir(OUTPUT_FOLDER)
    
    if os.path.isfile('output.txt'):
        os.remove('output.txt')

    # Read config file to get frame rate and calculate dt
    config = configparser.ConfigParser()
    config.read(os.path.join(TRAIN_FOLDER, 'ADL-Rundle-6', 'seqinfo.ini'))
    fps = float(config['Sequence']['frameRate'])
    dt = 1.0/fps
    width = int(config['Sequence']['imWidth'])
    height = int(config['Sequence']['imHeight'])

    gt = os.path.join(TRAIN_FOLDER, 'ADL-Rundle-6', 'gt', 'gt.txt')
    reader = FileReader(gt)
    fileWriter = FileWriter('output.txt')

    filters = []
    detections = []
    predictions = []

    # Initialize filters using the first frame (it should exists otherwise we are going crazy, right?!)
    initial_detection = reader.get_frame_content(1)
    for person in initial_detection:
        filters.append(KalmanFilter(8, dt, id=person[1]))
        filters[-1].update(np.array([person[2:6]]).T)
        fileWriter.writeLine(1, person[1], filters[-1].state[0, 0], filters[-1].state[1, 0], filters[-1].state[2, 0], filters[-1].state[3, 0])

    num_imgs = len(os.listdir(os.path.join(TRAIN_FOLDER, 'ADL-Rundle-6', 'img1')))
    writer = cv2.VideoWriter('tracking.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    for frame_id in range(2, num_imgs+1):
        filename = format_img_filename(frame_id, ".jpg")
        img = cv2.imread(os.path.join(TRAIN_FOLDER, 'ADL-Rundle-6', 'img1', filename))

        frame = reader.get_frame_content(frame_id)

        # Get a list containing all detections in current frame
        for person in frame:
            bb = person[1:6]
            detections.append(bb)
            cv2.rectangle(img, (int(bb[1]), int(bb[2])), (int(bb[1]+bb[3]), int(bb[2]+bb[4])), (0, 0, 255), 2)


        # Get another list containing the predictions for each filter considering the current frame
        for filter in filters:
            filter.predict(np.zeros((8, 1)))
            predictions.append(filter.state[0:4, 0])

        # Assign each bounding box detection to the respective filter using predictions to calculate
        # the most appropriate assignment
        for i in range(len(predictions)):
            # Calculate the IoU to get the best-fitting bounding box
            scores = [IoU(predictions[i], detections[j][1:]) for j in range(len(detections))]
            idx = np.argmax(scores)

            # If the best-fitting bounding IoU is better than a treshold, update the corresponding
            # filter tracking this bounding box.
            if scores[idx] > 0.6:
                filters[i].update(np.array([detections[idx][1:]]).T)
            detections[idx] = np.zeros(5)
        

        # Allocate new filters for remaining detections if applied
        for detection in detections:
            # If a detection is not null, then initialize a new filter for tracking it (probably a new object has entered the scene)
            if np.sum(detection) > 0.0:
                filters.append(KalmanFilter(8, dt, id=detection[0]))
                filters[-1].update(np.array([detection[1:]]).T)
            
        # Remove all filters that have not been up to date for a long time. For the remaining filters, draw trackings and save the results
        for filter in filters:
            if filter.downgrade_count > 4:
                filters.remove(filter)
            else:
                cv2.rectangle(img, (int(filter.state[0]), int(filter.state[1])), (int(filter.state[0]+filter.state[2]), int(filter.state[1]+filter.state[3])), (0, 255, 0), 2)
                cv2.putText(img, str(int(filter.id)), (int(filter.state[0]), int(filter.state[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                fileWriter.writeLine(frame_id, filter.id, filter.state[0, 0], filter.state[1, 0], filter.state[2, 0], filter.state[3, 0])
        
        writer.write(img)
        # cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), img)
        # cv2.imshow('frame '+str(frame_id), img)
        # cv2.waitKey(0)

        # Restore auxiliary variables 
        predictions = []
        detections = []

    writer.release()    

if __name__ == '__main__':
    main()