package modules.preprocessing;

import modules.models.ScaleRegion;
import org.opencv.core.Rect;
import java.util.ArrayList;
import java.util.List;

public class ImagePyramid {
    public List<ScaleRegion> generateRegions(int w, int h) {
        List<ScaleRegion> regions = new ArrayList<>();
        int overlap = 150;

        // DURCHLAUF 1: Standard-Größe (Zoom 1.0)
        regions.add(new ScaleRegion(new Rect(0, 0, w, h), 1.0));
        // DURCHLAUF 2: Künstlich Rauszoomen (Zoom 0.4)
        regions.add(new ScaleRegion(new Rect(0, 0, w, h), 0.4));

        // DURCHLAUF 3-6: SAHI Quadranten (Zoom-In für winzige Schilder)
        int halfW = w / 2;
        int halfH = h / 2;
        regions.add(new ScaleRegion(createSafeRect(0, 0, halfW + overlap, halfH + overlap, w, h), 1.0));
        regions.add(new ScaleRegion(createSafeRect(halfW - overlap, 0, w - (halfW - overlap), halfH + overlap, w, h), 1.0));
        regions.add(new ScaleRegion(createSafeRect(0, halfH - overlap, halfW + overlap, h - (halfH - overlap), w, h), 1.0));
        regions.add(new ScaleRegion(createSafeRect(halfW - overlap, halfH - overlap, w - (halfW - overlap), h - (halfH - overlap), w, h), 1.0));

        return regions;
    }

    private Rect createSafeRect(int x, int y, int width, int height, int maxW, int maxH) {
        int rx = Math.max(0, x);
        int ry = Math.max(0, y);
        int rw = Math.min(width, maxW - rx);
        int rh = Math.min(height, maxH - ry);
        return new Rect(rx, ry, rw, rh);
    }
}