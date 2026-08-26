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
        List<Rect2d> allBoxes = new ArrayList<>();
        List<Float> allScores = new ArrayList<>();
        List<Integer> allClassIds = new ArrayList<>();

        int w = image.cols();
        int h = image.rows();
        int overlap = 150;

        Rect fullImageRect = new Rect(0, 0, w, h);

        // --- DURCHLAUF 1: Standard-Größe (Zoom 1.0) ---
        runInference(image, fullImageRect, 1.0, confThreshold, allBoxes, allScores, allClassIds);

        // --- DURCHLAUF 2: Künstlich Rauszoomen (Zoom 0.5) für gigantische Schilder! ---
        runInference(image, fullImageRect, 0.4, confThreshold, allBoxes, allScores, allClassIds);

        // --- DURCHLAUF 3-6: SAHI Quadranten (Zoom-In für winzige Schilder) ---
        int halfW = w / 2;
        int halfH = h / 2;
        runInference(image, createSafeRect(0, 0, halfW + overlap, halfH + overlap, w, h), 1.0, confThreshold, allBoxes, allScores, allClassIds);
        runInference(image, createSafeRect(halfW - overlap, 0, w - (halfW - overlap), halfH + overlap, w, h), 1.0, confThreshold, allBoxes, allScores, allClassIds);
        runInference(image, createSafeRect(0, halfH - overlap, halfW + overlap, h - (halfH - overlap), w, h), 1.0, confThreshold, allBoxes, allScores, allClassIds);
        runInference(image, createSafeRect(halfW - overlap, halfH - overlap, w - (halfW - overlap), h - (halfH - overlap), w, h), 1.0, confThreshold, allBoxes, allScores, allClassIds);

        List<DetectionResult> finalResults = new ArrayList<>();
        if (allBoxes.isEmpty()) return finalResults;

        // --- POST-PROCESSING: NMS und IoM-Merging (Box-in-Box) ---
        MatOfRect2d boxesMat = new MatOfRect2d();
        boxesMat.fromList(allBoxes);
        MatOfFloat scoresMat = new MatOfFloat();
        scoresMat.fromList(allScores);
        MatOfInt indices = new MatOfInt();

        Dnn.NMSBoxes(boxesMat, scoresMat, confThreshold, nmsThreshold, indices);

        List<Integer> validIndices = new ArrayList<>();
        if (!indices.empty() && indices.rows() > 0) {
            for (int idx : indices.toArray()) {
                validIndices.add(idx);
            }
        }
        validIndices.sort((idx1, idx2) -> Float.compare(allScores.get(idx2), allScores.get(idx1)));

        boolean[] merged = new boolean[validIndices.size()];

        for (int i = 0; i < validIndices.size(); i++) {
            if (merged[i]) continue;

            int idx1 = validIndices.get(i);
            Rect2d baseBox = allBoxes.get(idx1);
            int classId = allClassIds.get(idx1);

            double finalLeft = baseBox.x;
            double finalTop = baseBox.y;
            double finalRight = baseBox.x + baseBox.width;
            double finalBottom = baseBox.y + baseBox.height;
            float finalScore = allScores.get(idx1);

            for (int j = i + 1; j < validIndices.size(); j++) {
                if (merged[j]) continue;

                int idx2 = validIndices.get(j);
                if (classId != allClassIds.get(idx2)) continue;

                Rect2d compareBox = allBoxes.get(idx2);

                double interLeft = Math.max(finalLeft, compareBox.x);
                double interTop = Math.max(finalTop, compareBox.y);
                double interRight = Math.min(finalRight, compareBox.x + compareBox.width);
                double interBottom = Math.min(finalBottom, compareBox.y + compareBox.height);

                if (interLeft < interRight && interTop < interBottom) {
                    double interArea = (interRight - interLeft) * (interBottom - interTop);
                    double area1 = (finalRight - finalLeft) * (finalBottom - finalTop);
                    double area2 = compareBox.width * compareBox.height;

                    if (interArea / Math.min(area1, area2) > 0.5) {
                        finalLeft = Math.min(finalLeft, compareBox.x);
                        finalTop = Math.min(finalTop, compareBox.y);
                        finalRight = Math.max(finalRight, compareBox.x + compareBox.width);
                        finalBottom = Math.max(finalBottom, compareBox.y + compareBox.height);
                        merged[j] = true;
                    }
                }
            }

            Rect2d finalBox = new Rect2d(finalLeft, finalTop, finalRight - finalLeft, finalBottom - finalTop);
            String name = "";
            if (classId == 7) name = "Vorfahrt Achten";
            if (classId == 21) name = "Vorfahrt";
            if (classId == 22) name = "Vorfahrtsstrasse";
            if (classId == 40) name = "Stopp";

            finalResults.add(new DetectionResult(finalBox, name, finalScore));
        }

        return finalResults;
    }

    private Rect createSafeRect(int x, int y, int width, int height, int maxW, int maxH) {
        int rx = Math.max(0, x);
        int ry = Math.max(0, y);
        int rw = Math.min(width, maxW - rx);
        int rh = Math.min(height, maxH - ry);
        return new Rect(rx, ry, rw, rh);
    }

    // --- NEU: Parameter zoomFactor hinzugefügt ---
    private void runInference(Mat fullImage, Rect cropRegion, double zoomFactor, float confThreshold,
                              List<Rect2d> outBoxes, List<Float> outScores, List<Integer> outClassIds) {

        Mat cropped = new Mat(fullImage, cropRegion);

        // Der künstliche Zoom wird einfach auf die Basis-Skalierung aufgerechnet!
        double scale = Math.min(640.0 / cropped.cols(), 640.0 / cropped.rows()) * zoomFactor;
        int newW = (int) Math.round(cropped.cols() * scale);
        int newH = (int) Math.round(cropped.rows() * scale);

        Mat resized = new Mat();
        Imgproc.resize(cropped, resized, new Size(newW, newH));

        Mat letterbox = new Mat(new Size(640, 640), cropped.type(), new Scalar(114, 114, 114));
        int left = (640 - newW) / 2;
        int top = (640 - newH) / 2;
        Mat roi = letterbox.submat(top, top + newH, left, left + newW);
        resized.copyTo(roi);

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
                double xCenter = data[index];
                double yCenter = data[index + 1];
                double boxW = data[index + 2];
                double boxH = data[index + 3];

                // Die Rückrechnung funktioniert automatisch fehlerfrei, da wir durch den modifizierten `scale` teilen!
                double cropXCenter = (xCenter - left) / scale;
                double cropYCenter = (yCenter - top) / scale;
                double cropBoxW = boxW / scale;
                double cropBoxH = boxH / scale;

                double cropLeft = cropXCenter - (cropBoxW / 2.0);
                double cropTop = cropYCenter - (cropBoxH / 2.0);

                double origLeft = cropLeft + cropRegion.x;
                double origTop = cropTop + cropRegion.y;

                outBoxes.add(new Rect2d(origLeft, origTop, cropBoxW, cropBoxH));
                outScores.add(maxScore);
                outClassIds.add(classId);
            }
        }
    }
}