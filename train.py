import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Parcours toutes les personnees dans le Training Set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Parcours toutes les images de chacun des personnes dans le Training Set
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                #S'il y a pas des personnes (ou plusieurs personnes) dans le training image, saute cette image
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                #Ajouter l'encodage pour l'actuelle image dans le trainign set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)


    #Determiner combien de voisins utiliser pour le classificateur KNN
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    #Creer et entrainer le classificateur KNN
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Enregistrer le model
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    #Charge le modele entrainé 
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Charger l'image et localiser la face
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # SI aucune image n'est trouvee dans l'image ==> retourne un resultat vide
    if len(X_face_locations) == 0:
        return []

    # Trouver les encodages pour les images dans le Training Set
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)


    # Utiliser le modele KNN pour trouver les meilleurs coincidances pour face de test
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Prediire les classes et enlever les classification qui ne sont pas dans le seuillage
    return [(pred, loc) if rec else ("12", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):

    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Dessing le rectangle en Utilisant le module PILLOW
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        name = name.encode("UTF-8")

        # Dessiner le label avec un nom en dessous
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Enlever le library de dessin de la memoire
    del draw

    # Afficher l'image
    pil_image.show()


if __name__ == "__main__":

    print("Training KNN classifier...")
    classifier = train("dataSet/train", model_save_path="model/trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
    # print ('Predicting...')
    # prediction = predict("dataSet/test", 'trained_knn_model.clf')
    # print('Prediction completed...')
