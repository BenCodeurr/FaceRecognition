import cv2
import pickle
from imutils.video import WebcamVideoStream
import face_recognition

class VideoCamera(object):
    def __init__(self):

        self.stream = WebcamVideoStream(src=0).start()
        with open("haarcascade_frontalface_default.xml", 'rb') as f:
            self.knn_clf = pickle.load(f)

    def __del__(self):
        self.stream.stop()

    def predict(self, frame, knn_clf, distance_threshold=0.4):
        # Find face locations
        X_face_locations = face_recognition.face_locations(frame)

        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        for i in range(len(X_face_locations)):
            print("closest_distances")
            print(closest_distances[0][i][0])

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("12", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    def get_frame(self):
        image = self.stream.read()

        predictions = self.predict(image, self.knn_clf)
        name = ''
        for name, (top, right, bottom, left) in predictions:
            startX = int(left)
            startY = int(top)
            endX = int(right)
            endY = int(bottom)

          #  try:
            cursor.execute("SELECT * FROM enfantb WHERE idEnfant = "+name)

            data = cursor.fetchall()
            nom = ''
            for row in data:
                    idE = row[0]
                    nom = row[1]
                    age = row[2]
                    sexe = row[3]
                    #print("id : ", idE, "Nom : ",nom)

            #except:
              #  print('Unable to fectch data')

            #db.close()

            #print("Voici le nom "+nom)
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 181, 51), 1)
            cv2.putText(image, nom, (endX - 70, endY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            #print(name)

        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        data.append(name)
        return data