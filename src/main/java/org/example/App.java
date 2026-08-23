package org.example;

import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

public class App {
    public static void main(String[] args) {
        // 1. OpenCV initialisieren (Lädt die nativen C++ Bindings)
        OpenCV.loadLocally();
        System.out.println("OpenCV geladen!");

        // 2. Das trainierte Modell laden
        String modelPath = "src/main/resources/models/best.onnx";
        Net net = Dnn.readNetFromONNX(modelPath);
        System.out.println("YOLO-Modell erfolgreich geladen!");

        // 3. Ein Testbild laden (Hier den Pfad zu einem echten Bild angeben!)
        // Lade dir am besten ein Bild von einem Stoppschild oder Vorfahrtsschild aus dem Internet herunter
        String imagePath = "C:\\Uni\\Bildverarbeitung\\VerkehrsschilderErkennung\\src\\main\\resources\\pictures\\vorfahrtAchten\\00430_jpg.rf.596d365481f10da537ead8ec4acafb02.jpg";
        Mat image = Imgcodecs.imread(imagePath);

        if (image.empty()) {
            System.out.println("Fehler: Konnte das Bild nicht finden. Pfad prüfen!");
            return;
        }

        // 4. Preprocessing: Bild für YOLOv8 vorbereiten
        // (Skaliert auf 640x640, normiert die Farbwerte auf 0.0 - 1.0, konvertiert BGR zu RGB)
        Mat blob = Dnn.blobFromImage(image, 1.0 / 255.0, new Size(640, 640), new Scalar(0), true, false);

        // 5. Inferenz: Bild durch das Netzwerk schicken
        net.setInput(blob);
        Mat output = net.forward();

        // 6. Post-Processing (Auswertung)
        Mat predictions = output.reshape(1, 47);
        Mat transposed = new Mat();
        org.opencv.core.Core.transpose(predictions, transposed);

        float[] data = new float[(int) transposed.total()];
        transposed.get(0, 0, data);

        int rows = transposed.rows(); // ca. 8400
        int cols = transposed.cols(); // 47
        float confidenceThreshold = 0.2f;

        // Listen zum Sammeln der Ergebnisse VOR der Filterung
        java.util.List<org.opencv.core.Rect2d> boxesList = new java.util.ArrayList<>();
        java.util.List<Float> scoresList = new java.util.ArrayList<>();
        java.util.List<Integer> classIdsList = new java.util.ArrayList<>();

        for (int i = 0; i < rows; i++) {
            int index = i * cols;

            float maxClassScore = 0;
            int classId = -1;

            for (int c = 4; c < cols; c++) {
                if (data[index + c] > maxClassScore) {
                    maxClassScore = data[index + c];
                    classId = c - 4;
                }
            }

            if (maxClassScore > confidenceThreshold) {
                if (classId == 7 || classId == 21 || classId == 22 || classId == 40) {
                    float x_center = data[index];
                    float y_center = data[index + 1];
                    float w = data[index + 2];
                    float h = data[index + 3];

                    // Umrechnung von Mitte auf Obere-Linke-Ecke für OpenCV
                    double left = x_center - (w / 2.0);
                    double top = y_center - (h / 2.0);

                    boxesList.add(new org.opencv.core.Rect2d(left, top, w, h));
                    scoresList.add(maxClassScore);
                    classIdsList.add(classId);
                }
            }
        }

        // 7. Blob immer zurück in ein anzeigbares Bild verwandeln
        java.util.List<Mat> unblobbed = new java.util.ArrayList<>();
        Dnn.imagesFromBlob(blob, unblobbed);
        Mat blobImage = new Mat();
        unblobbed.get(0).convertTo(blobImage, org.opencv.core.CvType.CV_8UC3, 255.0);
        Imgproc.cvtColor(blobImage, blobImage, Imgproc.COLOR_RGB2BGR);

        // 8. NMS und Zeichnen (Nur ausführen, wenn auch Boxen da sind)
        if (!boxesList.isEmpty()) {
            org.opencv.core.MatOfRect2d boxes = new org.opencv.core.MatOfRect2d();
            boxes.fromList(boxesList);

            org.opencv.core.MatOfFloat scores = new org.opencv.core.MatOfFloat();
            scores.fromList(scoresList);

            org.opencv.core.MatOfInt indices = new org.opencv.core.MatOfInt();
            float nmsThreshold = 0.45f;
            Dnn.NMSBoxes(boxes, scores, confidenceThreshold, nmsThreshold, indices);

            if (!indices.empty() && indices.rows() > 0) {
                int[] indicesArray = indices.toArray();
                double scaleX = (double) image.cols() / 640.0;
                double scaleY = (double) image.rows() / 640.0;

                for (int idx : indicesArray) {
                    org.opencv.core.Rect2d box = boxesList.get(idx);
                    int classId = classIdsList.get(idx);
                    float score = scoresList.get(idx);

                    String schildName = "";
                    if (classId == 7) schildName = "Vorfahrt Achten";
                    if (classId == 21) schildName = "Vorfahrt";
                    if (classId == 22) schildName = "Vorfahrtsstrasse";
                    if (classId == 40) schildName = "Stopp";

                    int x = (int) Math.round(box.x * scaleX);
                    int y = (int) Math.round(box.y * scaleY);
                    int w = (int) Math.round(box.width * scaleX);
                    int h = (int) Math.round(box.height * scaleY);

                    // Text-Hintergrund und Position berechnen
                    String label = String.format("%s (%.0f%%)", schildName, score * 100);
                    int[] baseLine = new int[1];
                    org.opencv.core.Size textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, 2, baseLine);
                    int textY = (y - 10 < textSize.height) ? y + (int)textSize.height + 10 : y - 10;

                    // Zeichnen NUR noch auf dem Originalbild (image)
                    Imgproc.rectangle(image, new org.opencv.core.Point(x, y), new org.opencv.core.Point(x + w, y + h), new org.opencv.core.Scalar(0, 255, 0), 3);
                    Imgproc.rectangle(image, new org.opencv.core.Point(x, textY - textSize.height - 5), new org.opencv.core.Point(x + textSize.width, textY + baseLine[0]), new org.opencv.core.Scalar(0, 0, 0), Imgproc.FILLED);
                    Imgproc.putText(image, label, new org.opencv.core.Point(x, textY), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new org.opencv.core.Scalar(255, 255, 255), 2);
                }
            } else {
                System.out.println("Kein Schild gefunden (durch NMS gefiltert).");
            }
        } else {
            System.out.println("Kein Schild gefunden (Score zu niedrig oder keines vorhanden).");
        }

        // 9. Fenster in jedem Fall anzeigen
        HighGui.imshow("Verkehrsschild Analyse - Original", image);
        HighGui.imshow("Verkehrsschild Analyse - Vorverarbeitung (640x640)", blobImage);
        HighGui.waitKey(0);
        System.exit(0);


    }
}