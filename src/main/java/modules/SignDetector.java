package modules;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class SignDetector {
    private final Net net;

    public SignDetector(String modelPath) {
        this.net = Dnn.readNetFromONNX(modelPath);
    }

    public List<DetectionResult> detect(Mat image, float confThreshold, float nmsThreshold) {
        List<DetectionResult> finalResults = new ArrayList<>();

        // 1. Pre-Processing: Letterboxing (Proportionen erhalten)
        double scale = Math.min(640.0 / image.cols(), 640.0 / image.rows());
        int newW = (int) Math.round(image.cols() * scale);
        int newH = (int) Math.round(image.rows() * scale);

        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(newW, newH));

        // Grauen 640x640 Hintergrund erstellen (YOLO Standardfarbe 114)
        Mat letterbox = new Mat(new Size(640, 640), image.type(), new Scalar(114, 114, 114));
        int left = (640 - newW) / 2;
        int top = (640 - newH) / 2;

        // Skaliertes Bild in die Mitte kopieren
        Mat roi = letterbox.submat(top, top + newH, left, left + newW);
        resized.copyTo(roi);

        // 2. Inferenz
        Mat blob = Dnn.blobFromImage(letterbox, 1.0 / 255.0, new Size(640, 640), new Scalar(0), true, false);
        net.setInput(blob);
        Mat output = net.forward();

        Mat predictions = output.reshape(1, 47);
        Mat transposed = new Mat();
        Core.transpose(predictions, transposed);

        float[] data = new float[(int) transposed.total()];
        transposed.get(0, 0, data);

        int rows = transposed.rows();
        int cols = transposed.cols();

        List<Rect2d> boxesList = new ArrayList<>();
        List<Float> scoresList = new ArrayList<>();
        List<Integer> classIdsList = new ArrayList<>();

        for (int i = 0; i < rows; i++) {
            int index = i * cols;
            float maxScore = 0;
            int classId = -1;

            for (int c = 4; c < cols; c++) {
                if (data[index + c] > maxScore) {
                    maxScore = data[index + c];
                    classId = c - 4;
                }
            }

            if (maxScore > confThreshold && (classId == 7 || classId == 21 || classId == 22 || classId == 40)) {
                // Koordinaten auf dem 640x640 Letterbox-Bild
                double xCenter = data[index];
                double yCenter = data[index + 1];
                double boxW = data[index + 2];
                double boxH = data[index + 3];

                // 3. Post-Processing: Geometrie zurück auf das Originalbild rechnen
                double origXCenter = (xCenter - left) / scale;
                double origYCenter = (yCenter - top) / scale;
                double origW = boxW / scale;
                double origH = boxH / scale;

                double origLeft = origXCenter - (origW / 2.0);
                double origTop = origYCenter - (origH / 2.0);

                boxesList.add(new Rect2d(origLeft, origTop, origW, origH));
                scoresList.add(maxScore);
                classIdsList.add(classId);
            }
        }

        if (boxesList.isEmpty()) return finalResults;

        MatOfRect2d boxes = new MatOfRect2d();
        boxes.fromList(boxesList);
        MatOfFloat scores = new MatOfFloat();
        scores.fromList(scoresList);
        MatOfInt indices = new MatOfInt();

        Dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);

        if (!indices.empty() && indices.rows() > 0) {
            for (int idx : indices.toArray()) {
                Rect2d box = boxesList.get(idx);
                int classId = classIdsList.get(idx);

                String name = "";
                if (classId == 7) name = "Vorfahrt Achten";
                if (classId == 21) name = "Vorfahrt naechste Kreuzung";
                if (classId == 22) name = "Vorfahrtsstrasse";
                if (classId == 40) name = "Stopp";

                finalResults.add(new DetectionResult(box, name, scoresList.get(idx)));
            }
        }
        return finalResults;
    }
}