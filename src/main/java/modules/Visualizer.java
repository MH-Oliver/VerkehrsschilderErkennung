package modules;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import javafx.scene.image.Image;
import java.io.ByteArrayInputStream;
import java.util.List;

public class Visualizer {
    public static Image processAndConvert(Mat image, List<DetectionResult> results) {
        Mat displayMat = image.clone();

        for (DetectionResult res : results) {
            Rect2d box = res.getBox();
            int x = (int) Math.round(box.x);
            int y = (int) Math.round(box.y);
            int w = (int) Math.round(box.width);
            int h = (int) Math.round(box.height);

            // 1. Rahmen zeichnen (Grün)
            Imgproc.rectangle(displayMat, new Point(x, y), new Point(x + w, y + h), new Scalar(0, 255, 0), 3);

            String label = String.format("%s (%.0f%%)", res.getClassName(), res.getScore() * 100);

            // 2. Größe des Textes berechnen
            int[] baseLine = new int[1];
            Size textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, 2, baseLine);

            // 3. Rand-Überprüfung Y-Achse (Oben)
            int textY;
            if (y - textSize.height - 15 < 0) {
                textY = y + (int) textSize.height + 10; // Schild berührt oberen Rand -> Text in die Box schieben
            } else {
                textY = y - 10; // Normal über der Box platzieren
            }

            // 4. Rand-Überprüfung X-Achse (Rechts & Links)
            int textX = x;
            if (textX + textSize.width > displayMat.cols()) {
                textX = displayMat.cols() - (int) textSize.width - 5; // Nach links verschieben
            }
            if (textX < 0) {
                textX = 5; // Am linken Rand absichern
            }

            // 5. Schwarzen Hintergrund und weißen Text zeichnen
            Imgproc.rectangle(displayMat,
                    new Point(textX, textY - textSize.height - 5),
                    new Point(textX + textSize.width, textY + baseLine[0]),
                    new Scalar(0, 0, 0), Imgproc.FILLED);

            Imgproc.putText(displayMat, label, new Point(textX, textY),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 2);
        }

        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", displayMat, buffer);
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }
}
