package modules;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class MainUI {
    private final Stage stage;
    private final SignDetector detector;
    private Mat currentImage;
    private final ImageView imageView;
    private final Slider confSlider;
    private final Slider nmsSlider;

    // --- NEU: Navigation State ---
    private List<File> currentDirFiles;
    private int currentFileIndex = -1;

    public MainUI(Stage stage) {
        this.stage = stage;
        this.detector = new SignDetector("src/main/resources/models/best.onnx");
        this.imageView = new ImageView();
        this.imageView.setPreserveRatio(true);
        this.imageView.setFitWidth(800);

        this.confSlider = new Slider(0, 1, 0.25);
        this.nmsSlider = new Slider(0, 1, 0.45);
    }

    public void buildAndShow() {
        Button loadBtn = new Button("Bild laden");
        loadBtn.setOnAction(e -> openFileChooser());

        Label confValueLabel = new Label(String.format("%.2f", confSlider.getValue()));
        Label nmsValueLabel = new Label(String.format("%.2f", nmsSlider.getValue()));

        confSlider.valueProperty().addListener((obs, oldV, newV) -> {
            confValueLabel.setText(String.format("%.2f", newV.doubleValue()));
            updateDetection();
        });
        nmsSlider.valueProperty().addListener((obs, oldV, newV) -> {
            nmsValueLabel.setText(String.format("%.2f", newV.doubleValue()));
            updateDetection();
        });

        HBox confHeader = new HBox(10, new Label("Confidence:"), confValueLabel);
        HBox nmsHeader = new HBox(10, new Label("NMS Threshold:"), nmsValueLabel);

        VBox controls = new VBox(15, loadBtn, confHeader, confSlider, nmsHeader, nmsSlider);
        controls.setPadding(new Insets(20));
        controls.setPrefWidth(260);

        BorderPane root = new BorderPane();
        root.setLeft(controls);
        root.setCenter(imageView);

        Scene scene = new Scene(root, 1100, 700);

        // --- NEU: Globale Tastatur-Überwachung ---
        scene.addEventFilter(KeyEvent.KEY_PRESSED, e -> handleKeyPress(e));

        stage.setTitle("Verkehrsschild Analyse Tool");
        stage.setScene(scene);
        stage.show();
    }

    private void handleKeyPress(KeyEvent e) {
        if (currentDirFiles == null || currentDirFiles.isEmpty()) return;

        if (e.getCode() == KeyCode.UP || e.getCode() == KeyCode.LEFT) {
            currentFileIndex = (currentFileIndex - 1 >= 0) ? currentFileIndex - 1 : currentDirFiles.size() - 1;
            loadFileAsBytes(currentDirFiles.get(currentFileIndex));
            e.consume(); // Verhindert, dass Slider auf die Tasten reagieren
        } else if (e.getCode() == KeyCode.DOWN || e.getCode() == KeyCode.RIGHT) {
            currentFileIndex = (currentFileIndex + 1 < currentDirFiles.size()) ? currentFileIndex + 1 : 0;
            loadFileAsBytes(currentDirFiles.get(currentFileIndex));
            e.consume();
        }
    }

    private void openFileChooser() {
        FileChooser chooser = new FileChooser();
        File initDir = new File("src/main/resources/pictures");
        if (initDir.exists()) chooser.setInitialDirectory(initDir);

        File file = chooser.showOpenDialog(stage);
        if (file != null) {
            updateDirectoryFiles(file);
            loadFileAsBytes(file);
        }
    }

    private void updateDirectoryFiles(File currentFile) {
        File dir = currentFile.getParentFile();
        if (dir != null && dir.isDirectory()) {
            File[] files = dir.listFiles();
            if (files != null) {
                // Nur Bilder in die Liste aufnehmen und sortieren
                currentDirFiles = Arrays.stream(files)
                        .filter(f -> f.getName().toLowerCase().matches(".*\\.(png|jpg|jpeg)"))
                        .sorted()
                        .collect(Collectors.toList());
                currentFileIndex = currentDirFiles.indexOf(currentFile);
            }
        }
    }

    private void loadFileAsBytes(File file) {
        try {
            byte[] fileContent = Files.readAllBytes(file.toPath());
            MatOfByte buffer = new MatOfByte(fileContent);
            currentImage = Imgcodecs.imdecode(buffer, Imgcodecs.IMREAD_COLOR);

            if (!currentImage.empty()) {
                updateDetection();
            }
        } catch (Exception ex) {
            System.out.println("Fehler beim Laden: " + ex.getMessage());
        }
    }

    private void updateDetection() {
        if (currentImage == null || currentImage.empty()) return;
        List<DetectionResult> results = detector.detect(currentImage, (float) confSlider.getValue(), (float) nmsSlider.getValue());
        imageView.setImage(Visualizer.processAndConvert(currentImage, results));
    }
}