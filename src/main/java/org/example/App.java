package org.example;

import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

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
        String imagePath = "C:\\Uni\\Bildverarbeitung\\VerkehrsschilderErkennung\\src\\main\\resources\\testbild.jpg";
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
        // Matrix umformen: Aus 3D machen wir eine 2D-Tabelle (8400 Zeilen, 47 Spalten)
        Mat predictions = output.reshape(1, 47);
        Mat transposed = new Mat();
        org.opencv.core.Core.transpose(predictions, transposed);

        // Alle Daten effizient in ein Java-Float-Array laden
        float[] data = new float[(int) transposed.total()];
        transposed.get(0, 0, data);

        int rows = transposed.rows(); // ca. 8400
        int cols = transposed.cols(); // 47
        float confidenceThreshold = 0.6f; // Mindestens 60% Sicherheit

        System.out.println("Durchsuche " + rows + " mögliche Boxen...");

        for (int i = 0; i < rows; i++) {
            int index = i * cols;

            float maxClassScore = 0;
            int classId = -1;

            // Finde die Klasse mit dem höchsten Score (Werte starten ab Index 4)
            for (int c = 4; c < cols; c++) {
                if (data[index + c] > maxClassScore) {
                    maxClassScore = data[index + c];
                    classId = c - 4; // -4, da die ersten 4 Werte die Koordinaten sind
                }
            }

            // Wenn das Netz sicher ist und es eines unserer 4 Schilder ist
            if (maxClassScore > confidenceThreshold) {
                if (classId == 7 || classId == 21 || classId == 22 || classId == 40) {
                    float x = data[index];
                    float y = data[index + 1];
                    float w = data[index + 2];
                    float h = data[index + 3];

                    String schildName = "";
                    if (classId == 7) schildName = "Vorfahrt Achten";
                    if (classId == 21) schildName = "Vorfahrt";
                    if (classId == 22) schildName = "Vorfahrtsstrasse";
                    if (classId == 40) schildName = "Stopp";

                    System.out.println("\n--- SCHILD ERKANNT ---");
                    System.out.println("Typ: " + schildName + " (Score: " + maxClassScore + ")");
                    System.out.println("Box: Mitte(" + x + ", " + y + "), Breite=" + w + ", Hoehe=" + h);
                }
            }
        }
    }
}