package modules;

import javafx.application.Application;
import javafx.stage.Stage;
import nu.pattern.OpenCV;

public class App extends Application {
    public static void main(String[] args) {
        // Lädt die nativen OpenCV-Bibliotheken einmalig beim Start
        OpenCV.loadLocally();
        // Startet das JavaFX Framework
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        // Initialisiert und öffnet das Setup-Fenster
        new MainUI(primaryStage).buildAndShow();
    }
}