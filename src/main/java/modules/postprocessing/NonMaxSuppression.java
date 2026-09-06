package modules.postprocessing;

import modules.models.DetectionResult;
import modules.models.RawDetection;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.dnn.Dnn;
import java.util.ArrayList;
import java.util.List;

/**
 * Führt das Post-Processing der Detektionen durch, um redundante und
 * überlappende Vorhersagen desselben Objekts zu entfernen.
 */
public class NonMaxSuppression {

    /**
     * Filtert die rohen Detektionen über NMS und ein eigenes Box-Merging.
     * <p>
     * Schritte:
     * 1. Führt die Standard-OpenCV Non-Maximum Suppression (NMS) aus.
     * 2. Sortiert die verbliebenen Boxen absteigend nach ihrem Score (Konfidenz).
     * 3. Durchläuft die Boxen paarweise und prüft auf starke Überlappung (Box-in-Box / IoM).
     * 4. Verschmilzt überlappende Boxen derselben Klasse zu einer gemeinsamen Bounding Box.
     * 5. Übersetzt die numerischen Klassen-IDs in lesbare Text-Labels.
     *
     * @param rawDetections Liste der Vorhersagen aus dem CNN
     * @param confThreshold Konfidenz-Schwellwert
     * @param nmsThreshold Schwellwert für die NMS-Überlappung
     * @return Bereinigte Liste der finalen Detektionen
     */
    public List<DetectionResult> filter(List<RawDetection> rawDetections, float confThreshold, float nmsThreshold) {
        List<DetectionResult> finalResults = new ArrayList<>();
        if (rawDetections.isEmpty()) return finalResults;

        List<Rect2d> allBoxes = new ArrayList<>();
        List<Float> allScores = new ArrayList<>();
        for (RawDetection d : rawDetections) {
            allBoxes.add(d.box);
            allScores.add(d.score);
        }

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
            RawDetection base = rawDetections.get(idx1);
            double finalLeft = base.box.x;
            double finalTop = base.box.y;
            double finalRight = base.box.x + base.box.width;
            double finalBottom = base.box.y + base.box.height;

            for (int j = i + 1; j < validIndices.size(); j++) {
                if (merged[j]) continue;
                int idx2 = validIndices.get(j);
                RawDetection compare = rawDetections.get(idx2);
                if (base.classId != compare.classId) continue;

                double interLeft = Math.max(finalLeft, compare.box.x);
                double interTop = Math.max(finalTop, compare.box.y);
                double interRight = Math.min(finalRight, compare.box.x + compare.box.width);
                double interBottom = Math.min(finalBottom, compare.box.y + compare.box.height);

                if (interLeft < interRight && interTop < interBottom) {
                    double interArea = (interRight - interLeft) * (interBottom - interTop);
                    double area1 = (finalRight - finalLeft) * (finalBottom - finalTop);
                    double area2 = compare.box.width * compare.box.height;

                    if (interArea / Math.min(area1, area2) > 0.5) {
                        finalLeft = Math.min(finalLeft, compare.box.x);
                        finalTop = Math.min(finalTop, compare.box.y);
                        finalRight = Math.max(finalRight, compare.box.x + compare.box.width);
                        finalBottom = Math.max(finalBottom, compare.box.y + compare.box.height);
                        merged[j] = true;
                    }
                }
            }
            Rect2d finalBox = new Rect2d(finalLeft, finalTop, finalRight - finalLeft, finalBottom - finalTop);
            finalResults.add(new DetectionResult(finalBox, getClassName(base.classId), base.score));
        }
        return finalResults;
    }

    private String getClassName(int classId) {
        if (classId == 7) return "Vorfahrt Achten";
        if (classId == 21) return "Vorfahrt";
        if (classId == 22) return "Vorfahrtsstrasse";
        if (classId == 40) return "Stopp";
        return "Unbekannt";
    }
}