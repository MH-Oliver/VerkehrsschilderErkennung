package modules;

import javafx.geometry.Bounds;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.*;
import javafx.scene.shape.Rectangle;
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

    private List<File> currentDirFiles;
    private int currentFileIndex = -1;

    public MainUI(Stage stage) {
        this.stage = stage;
        this.detector = new SignDetector("src/main/resources/models/best_02.onnx");

        this.imageView = new ImageView();
        this.imageView.setPreserveRatio(true);
        this.imageView.setFitWidth(800);
        this.imageView.setFitHeight(600);

        // --- AKTUALISIERT: Zoom auf Mausposition mit Limit ---
        this.imageView.setOnScroll((ScrollEvent event) -> {
            double oldScale = imageView.getScaleX();
            double zoomFactor = 1.1;

            if (event.getDeltaY() < 0) {
                zoomFactor = 1 / zoomFactor; // Rauszoomen
            }

            double newScale = oldScale * zoomFactor;

            // Verhindern, dass man kleiner als die Originalgröße zoomt
            if (newScale <= 1.0) {
                imageView.setScaleX(1.0);
                imageView.setScaleY(1.0);
                imageView.setTranslateX(0);
                imageView.setTranslateY(0);
                event.consume();
                return;
            }

            // Mathematik für den Zoom auf die exakte Mausposition
            double f = (newScale / oldScale) - 1;

            Bounds bounds = imageView.localToScene(imageView.getBoundsInLocal());
            double dx = (event.getSceneX() - (bounds.getWidth() / 2 + bounds.getMinX()));
            double dy = (event.getSceneY() - (bounds.getHeight() / 2 + bounds.getMinY()));

            // Bild exakt gegen die Vergrößerung verschieben
            imageView.setTranslateX(imageView.getTranslateX() - f * dx);
            imageView.setTranslateY(imageView.getTranslateY() - f * dy);

            imageView.setScaleX(newScale);
            imageView.setScaleY(newScale);

            event.consume();
        });

        this.confSlider = new Slider(0, 1, 0.25);
        this.nmsSlider = new Slider(0, 1, 0.45);
    }

    public void buildAndShow() {
        Button loadBtn = new Button("Bild laden");
        loadBtn.setOnAction(e -> openFileChooser());

        Button resetZoomBtn = new Button("Zoom Reset");
        resetZoomBtn.setOnAction(e -> {
            imageView.setScaleX(1.0);
            imageView.setScaleY(1.0);
            imageView.setTranslateX(0);
            imageView.setTranslateY(0);
        });

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
        HBox buttonBox = new HBox(10, loadBtn, resetZoomBtn);

        VBox controls = new VBox(15, buttonBox, confHeader, confSlider, nmsHeader, nmsSlider);
        controls.setPadding(new Insets(20));
        controls.setPrefWidth(280);

        // --- AKTUALISIERT: Clipping, damit das Bild nicht über das Menü lappt ---
        Pane imageContainer = new Pane(imageView);
        Rectangle clip = new Rectangle();
        clip.widthProperty().bind(imageContainer.widthProperty());
        clip.heightProperty().bind(imageContainer.heightProperty());
        imageContainer.setClip(clip);

        BorderPane root = new BorderPane();
        root.setLeft(controls);
        root.setCenter(imageContainer);

        Scene scene = new Scene(root, 1100, 700);
        scene.addEventFilter(KeyEvent.KEY_PRESSED, this::handleKeyPress);

        stage.setTitle("Verkehrsschild Analyse Tool");
        stage.setScene(scene);
        stage.show();
    }

    private void handleKeyPress(KeyEvent e) {
        if (currentDirFiles == null || currentDirFiles.isEmpty()) return;

        if (e.getCode() == KeyCode.UP || e.getCode() == KeyCode.LEFT) {
            currentFileIndex = (currentFileIndex - 1 >= 0) ? currentFileIndex - 1 : currentDirFiles.size() - 1;
            loadFileAsBytes(currentDirFiles.get(currentFileIndex));
            e.consume();
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
                // Zoom & Translation bei jedem neuen Bild sicherheitshalber zurücksetzen
                imageView.setScaleX(1.0);
                imageView.setScaleY(1.0);
                imageView.setTranslateX(0);
                imageView.setTranslateY(0);
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