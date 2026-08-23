package modules;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.File;
import java.nio.file.Files;
import java.util.List;

public class MainUI {
    private final Stage stage;
    private final SignDetector detector;
    private Mat currentImage;
    private final ImageView imageView;
    private final Slider confSlider;
    private final Slider nmsSlider;

    public MainUI(Stage stage) {
        this.stage = stage;
        this.detector = new SignDetector("src/main/resources/models/best.onnx");
        this.imageView = new ImageView();
        this.imageView.setPreserveRatio(true);
        this.imageView.setFitWidth(800); // Bild an Fenstergröße anpassen

        // Slider mit Standardwerten (25% Confidence, 45% NMS)
        this.confSlider = new Slider(0, 1, 0.25);
        this.nmsSlider = new Slider(0, 1, 0.45);
    }

    public void buildAndShow() {
        Button loadBtn = new Button("Bild laden");
        loadBtn.setOnAction(e -> loadImage());

        // Labels für die aktuellen Werte erstellen (mit 2 Nachkommastellen formatiert)
        Label confValueLabel = new Label(String.format("%.2f", confSlider.getValue()));
        Label nmsValueLabel = new Label(String.format("%.2f", nmsSlider.getValue()));

        // Listener anpassen, sodass sich der Text beim Schieben sofort aktualisiert
        confSlider.valueProperty().addListener((obs, oldV, newV) -> {
            confValueLabel.setText(String.format("%.2f", newV.doubleValue()));
            updateDetection();
        });
        nmsSlider.valueProperty().addListener((obs, oldV, newV) -> {
            nmsValueLabel.setText(String.format("%.2f", newV.doubleValue()));
            updateDetection();
        });

        // Layout: Text und Wert nebeneinander packen
        HBox confHeader = new HBox(10, new Label("Confidence Threshold:"), confValueLabel);
        HBox nmsHeader = new HBox(10, new Label("NMS Threshold:"), nmsValueLabel);

        VBox controls = new VBox(15, loadBtn, confHeader, confSlider, nmsHeader, nmsSlider);
        controls.setPadding(new Insets(20));
        controls.setPrefWidth(260);

        BorderPane root = new BorderPane();
        root.setLeft(controls);
        root.setCenter(imageView);

        stage.setTitle("Verkehrsschild Analyse Tool");
        stage.setScene(new Scene(root, 1100, 700));
        stage.show();
    }

    private void loadImage() {
        FileChooser chooser = new FileChooser();
        File initDir = new File("src/main/resources/pictures");
        if (initDir.exists()) chooser.setInitialDirectory(initDir);

        File file = chooser.showOpenDialog(stage);
        if (file != null) {
            try {
                // 1. Java liest die Datei als rohes Byte-Array ein (ignoriert Sonderzeichen-Probleme)
                byte[] fileContent = Files.readAllBytes(file.toPath());

                // 2. Wir packen die Bytes in einen OpenCV-Datencontainer
                MatOfByte buffer = new MatOfByte(fileContent);

                // 3. OpenCV dekodiert das Bild direkt aus dem Arbeitsspeicher
                currentImage = Imgcodecs.imdecode(buffer, Imgcodecs.IMREAD_COLOR);

                if (currentImage.empty()) {
                    System.out.println("Fehler: Das Bild konnte nicht dekodiert werden.");
                    return;
                }

                updateDetection();

            } catch (Exception ex) {
                System.out.println("Fehler beim Laden der Datei: " + ex.getMessage());
            }
        }
    }

    private void updateDetection() {
        if (currentImage == null || currentImage.empty()) return;
        List<DetectionResult> results = detector.detect(currentImage, (float) confSlider.getValue(), (float) nmsSlider.getValue());
        imageView.setImage(Visualizer.processAndConvert(currentImage, results));
    }
}