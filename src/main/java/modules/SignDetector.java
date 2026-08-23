package modules;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import java.util.ArrayList;
import java.util.List;

public class SignDetector {
    private final Net net;

    public SignDetector(String modelPath) {
        this.net = Dnn.readNetFromONNX(modelPath);
    }

    public List<DetectionResult> detect(Mat image, float confThreshold, float nmsThreshold) {
        List<DetectionResult> finalResults = new ArrayList<>();

        Mat blob = Dnn.blobFromImage(image, 1.0 / 255.0, new Size(640, 640), new Scalar(0), true, false);
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
                double left = data[index] - (data[index + 2] / 2.0);
                double top = data[index + 1] - (data[index + 3] / 2.0);
                boxesList.add(new Rect2d(left, top, data[index + 2], data[index + 3]));
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
            double scaleX = (double) image.cols() / 640.0;
            double scaleY = (double) image.rows() / 640.0;

            for (int idx : indices.toArray()) {
                Rect2d box = boxesList.get(idx);
                int classId = classIdsList.get(idx);

                String name = "";
                if (classId == 7) name = "Vorfahrt Achten";
                if (classId == 21) name = "Vorfahrt naechste Kreuzung";
                if (classId == 22) name = "Vorfahrtsstrasse";
                if (classId == 40) name = "Stopp";

                Rect2d scaledBox = new Rect2d(box.x * scaleX, box.y * scaleY, box.width * scaleX, box.height * scaleY);
                finalResults.add(new DetectionResult(scaledBox, name, scoresList.get(idx)));
            }
        }
        return finalResults;
    }
}