import numpy as np 
import os

class FileReader():
    def __init__(self, filename):
        self.filename = filename
        self.contents = np.loadtxt(filename, delimiter=',')

    def sort(self, columns):
        for column in columns:
            self.contents = self.contents[self.contents[:, column].argsort(kind='stable')]

    def save(self, filename):
        np.savetxt(filename, self.contents, fmt='%d')

    def get_frame_content(self, frame_id):
        return self.contents[self.contents[:, 0] == frame_id, :]

class FileWriter():
    def __init__(self, filename):
        self.filename = filename

    def writeLine(self, frame_id, id, bb_left, bb_top, width, height, confidence=-1, x=-1, y=-1, z=-1):
        mode = 'a' if os.path.isfile(self.filename) else 'w'
        with open(self.filename, mode) as f:
            line = str(int(frame_id))+','+str(int(id))+','+'{:.2f}'.format(bb_left)+','+'{:.2f}'.format(bb_top)+','+'{:.2f}'.format(width)+','+'{:.2f}'.format(height)+','+str(int(confidence))+','+str(int(x))+','+str(int(y))+','+str(int(z))+'\n'
            f.write(line)
            f.close()

class KalmanFilter():
    def __init__(self, num_variables, dt, id=-1):
        # Object id
        self.id = id

        # Internal variable to control the filter lifetime
        self.downgrade_count = 0

        # State: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.zeros((num_variables, 1))

        # Covariance matrix
        self.covariance = 100*np.eye(num_variables, num_variables)

        self.control_matrix = np.zeros((num_variables, num_variables)) # does not influence my model

        # Transition matriz: each prediction is a linear model of the respective input
        # var = var + vvar*dt
        self.transition_matrix = np.eye(num_variables) # A
        self.transition_matrix[0, 4] = dt
        self.transition_matrix[1, 5] = dt
        self.transition_matrix[2, 6] = dt
        self.transition_matrix[3, 7] = dt

        self.measurement_noise = np.eye(int(num_variables/2)) # Q
        self.measurement_noise[-1, -1] = 10
        self.measurement_noise[-2, -2] = 10

        self.covariance_matrix = 0.01*np.eye(num_variables) # R

        # Measurement matrix: describes what state variables are measured from the environment
        # In this case, we receive the position, width and height of bounding boxes
        self.measurement_matrix = np.eye(int(num_variables/2), num_variables) # C

    def predict(self, control_signal):
        self.state = self.transition_matrix @ self.state + self.control_matrix @ control_signal
        self.covariance = self.transition_matrix @ self.covariance @ self.transition_matrix.T + self.covariance_matrix
        self.downgrade_count += 1

    def update(self, measurement):
        kalman_gain = self.covariance @ self.measurement_matrix.T @ np.linalg.inv(self.measurement_matrix @ self.covariance @ self.measurement_matrix.T + self.measurement_noise)
        self.state = self.state + kalman_gain @ (measurement-self.measurement_matrix @ self.state)
        self.covariance = (np.eye(self.covariance.shape[0]) - kalman_gain @ self.measurement_matrix) @ self.covariance
        self.downgrade_count = 0

