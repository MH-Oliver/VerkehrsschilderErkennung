package modules.pipeline;

import modules.models.DetectionResult;
import modules.models.RawDetection;
import modules.models.ScaleRegion;
import modules.preprocessing.ImagePyramid;
import modules.inference.ConvolutionalNeuralNet;
import modules.postprocessing.NonMaxSuppression;
import org.opencv.core.Mat;
import java.util.List;

/**
 * Orchestriert die gesamte Pipeline der Objekterkennung.
 */
public class ObjectDetector {
    private final ConvolutionalNeuralNet cnn;
    private final ImagePyramid pyramid;
    private final NonMaxSuppression nms;

    public ObjectDetector(String modelPath) {
        this.cnn = new ConvolutionalNeuralNet(modelPath);
        this.pyramid = new ImagePyramid();
        this.nms = new NonMaxSuppression();
    }

    /**
     * Führt den kompletten Erkennungsprozess für ein Bild durch.
     * <p>
     * Schritte:
     * 1. Preprocessing: Erzeugt die Skalierungsstufen (Bildpyramide).
     * 2. Inferenz: Wendet das YOLO-Netzwerk auf alle Stufen an.
     * 3. Postprocessing: Filtert redundante Vorhersagen.
     *
     * @param image Das zu analysierende Eingabebild
     * @param confThreshold Konfidenz-Schwellwert
     * @param nmsThreshold NMS-Schwellwert
     * @return Liste der finalen Verkehrszeichen
     */
    public List<DetectionResult> detect(Mat image, float confThreshold, float nmsThreshold) {
        // 1. Preprocessing: Regions definieren
        List<ScaleRegion> regions = pyramid.generateRegions(image.cols(), image.rows());

        // 2. Inference: YOLO-Netzwerk anwenden
        List<RawDetection> rawDetections = cnn.runInference(image, regions, confThreshold);

        // 3. Postprocessing: NMS und Merging durchführen
        return nms.filter(rawDetections, confThreshold, nmsThreshold);
    }
}