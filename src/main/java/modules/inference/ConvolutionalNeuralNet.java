package modules.inference;

import modules.models.RawDetection;
import modules.models.ScaleRegion;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

/**
 * Führt die eigentliche Objekterkennung mithilfe eines YOLO-basierten
 * Deep Convolutional Neural Networks durch.
 */
public class ConvolutionalNeuralNet {
    private final Net net;

    public ConvolutionalNeuralNet(String modelPath) {
        this.net = Dnn.readNetFromONNX(modelPath);
    }

    public List<RawDetection> runInference(Mat fullImage, List<ScaleRegion> regions, float confThreshold) {
        List<RawDetection> allDetections = new ArrayList<>();
        for (ScaleRegion region : regions) {
            processRegion(fullImage, region, confThreshold, allDetections);
        }
        return allDetections;
    }

    /**
     * Führt die YOLO-Vorhersage für eine spezifische Bildregion durch.
     * <p>
     * Schritte:<p>
     * 1. Schneidet die Region aus dem Originalbild aus.<p>
     * 2. Berechnet den finalen Skalierungsfaktor und verkleinert den Ausschnitt.<p>
     * 3. Erstellt eine quadratische 640x640 "Letterbox" mit grauem Rand (YOLO-Format).<p>
     * 4. Wandelt das Bild in einen Blob um und führt den Forward-Pass (Inferenz) aus.<p>
     * 5. Durchläuft den Netz-Output und filtert Vorhersagen über dem Threshold.<p>
     * 6. Rechnet die Koordinaten aus dem 640x640-Raster zurück in Originalbild-Koordinaten.
     */
    private void processRegion(Mat fullImage, ScaleRegion region, float confThreshold, List<RawDetection> detections) {
        Rect cropRegion = region.rect;
        Mat cropped = new Mat(fullImage, cropRegion);

        double scale = Math.min(640.0 / cropped.cols(), 640.0 / cropped.rows()) * region.zoomFactor;
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
                double cropXCenter = (data[index] - left) / scale;
                double cropYCenter = (data[index + 1] - top) / scale;
                double cropBoxW = data[index + 2] / scale;
                double cropBoxH = data[index + 3] / scale;

                double origLeft = (cropXCenter - (cropBoxW / 2.0)) + cropRegion.x;
                double origTop = (cropYCenter - (cropBoxH / 2.0)) + cropRegion.y;

                detections.add(new RawDetection(new Rect2d(origLeft, origTop, cropBoxW, cropBoxH), maxScore, classId));
            }
        }
    }
}