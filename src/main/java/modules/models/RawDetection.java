package modules.models;
import org.opencv.core.Rect2d;

public class RawDetection {
    public final Rect2d box;
    public final float score;
    public final int classId;

    public RawDetection(Rect2d box, float score, int classId) {
        this.box = box;
        this.score = score;
        this.classId = classId;
    }
}
