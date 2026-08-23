package modules;

import org.opencv.core.Rect2d;

public class DetectionResult {
    private final Rect2d box;
    private final String className;
    private final float score;

    public DetectionResult(Rect2d box, String className, float score) {
        this.box = box;
        this.className = className;
        this.score = score;
    }

    public Rect2d getBox() { return box; }
    public String getClassName() { return className; }
    public float getScore() { return score; }
}