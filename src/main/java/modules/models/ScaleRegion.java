package modules.models;
import org.opencv.core.Rect;

public class ScaleRegion {
    public final Rect rect;
    public final double zoomFactor;

    public ScaleRegion(Rect rect, double zoomFactor) {
        this.rect = rect;
        this.zoomFactor = zoomFactor;
    }
}