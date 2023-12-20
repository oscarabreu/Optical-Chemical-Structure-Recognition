import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class ImageConvert {

    private static final String IMAGE_PATH_PNG = "C:\\Users\\Administrator\\Desktop\\caffeine1.png";
    private static final String IMAGE_PATH_JPEG_NEW = "C:\\Users\\Administrator\\Desktop\\caffeine2.jpg";
    private static final String IMAGE_PATH_PNG_NEW = "F:\\Gis开发\\学习资料\\tile\\3_0_1_new.png";
    private static final String IMAGE_PATH_JPEG = "F:\\Gis开发\\学习资料\\tile\\3_0_1.jpg";


    public static void main(String[] args) {
        png2jpeg();
    }

    public static void png2jpeg() {
        //读取图片
        FileOutputStream fos = null;
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(IMAGE_PATH_PNG));
            //转成jpeg、
            BufferedImage bufferedImage1 = new BufferedImage(bufferedImage.getWidth(),
                    bufferedImage.getHeight(),
                    BufferedImage.TYPE_INT_RGB);
            bufferedImage1.createGraphics().drawImage(bufferedImage, 0, 0, Color.white, null);
            fos = new FileOutputStream(IMAGE_PATH_JPEG_NEW);
            ImageIO.write(bufferedImage, "jpg", fos);
            fos.flush();
        } catch (IOException e) {
            e.printStackTrace();
            try {
                fos.close();
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        }
    }

    public static void jpeg2png() {
        //读取图片
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(IMAGE_PATH_JPEG));
            //转成png、
            BufferedImage bufferedImage1 = new BufferedImage(bufferedImage.getWidth(),
                    bufferedImage.getHeight(),
                    BufferedImage.TYPE_INT_ARGB);
            bufferedImage1.createGraphics().drawImage(bufferedImage, 0, 0, Color.white, null);
            FileOutputStream fos = new FileOutputStream(IMAGE_PATH_PNG_NEW);
            ImageIO.write(bufferedImage1, "png", fos);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}