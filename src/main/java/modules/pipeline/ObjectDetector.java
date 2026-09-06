package modules.pipeline;

import modules.models.DetectionResult;
import modules.models.RawDetection;
import modules.models.ScaleRegion;
import modules.preprocessing.ImagePyramid;
import modules.inference.ConvolutionalNeuralNet;
import modules.postprocessing.NonMaxSuppression;
import org.opencv.core.Mat;
import java.util.List;

public class ObjectDetector {
    private final ConvolutionalNeuralNet cnn;
    private final ImagePyramid pyramid;
    private final NonMaxSuppression nms;

    public ObjectDetector(String modelPath) {
        this.cnn = new ConvolutionalNeuralNet(modelPath);
        this.pyramid = new ImagePyramid();
        this.nms = new NonMaxSuppression();
    }

    public List<DetectionResult> detect(Mat image, float confThreshold, float nmsThreshold) {
        // 1. Preprocessing: Regions definieren
        List<ScaleRegion> regions = pyramid.generateRegions(image.cols(), image.rows());

        // 2. Inference: YOLO-Netzwerk anwenden
        List<RawDetection> rawDetections = cnn.runInference(image, regions, confThreshold);

        // 3. Postprocessing: NMS und Merging durchführen
        return nms.filter(rawDetections, confThreshold, nmsThreshold);
    }
}