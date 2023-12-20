import org.openscience.cdk.exception.CDKException;
import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ReplaceSubstituentCanonicalAsterisk {
    public static void main(String[] args) throws IOException, CDKException {
        String fileRead = "125wanKS_Canonical.txt";
        String fileWrite = "125wanKS_Final.txt";
        ReplaceStarWithRandomHalogen(fileRead, fileWrite);
    }

    public static void ReplaceStarWithRandomHalogen(String fileRead, String fileWrite) throws CDKException, IOException {
        File record = new File(fileWrite);
        FileWriter writer = new FileWriter(record, true);
        File file = new File(fileRead);
        BufferedReader reader = null;
        String temp = null;
        List<String> halogens = Arrays.asList("F", "Cl", "Br", "I");

        try {
            reader = new BufferedReader(new FileReader(file));
            while ((temp = reader.readLine()) != null) {
                String[] aa = temp.split("\t");
                String originalPart = aa[0]; // The part before '\t'
                temp = aa[1];

                // Define a pattern to find asterisks
                Pattern asteriskPattern = Pattern.compile("\\*");
                Matcher matcher = asteriskPattern.matcher(temp);

                // Replace each asterisk with a random halogen
                StringBuilder new_0 = new StringBuilder();
                int lastIndex = 0;
                Random random = new Random();
                while (matcher.find()) {
                    int startIndex = matcher.start();
                    new_0.append(temp, lastIndex, startIndex);
                    // Get a random halogen from the list
                    String randomHalogen = halogens.get(random.nextInt(halogens.size()));
                    new_0.append(randomHalogen);
                    lastIndex = matcher.end();
                }
                new_0.append(temp.substring(lastIndex));

                // Rebuild the line with the modified new_0
                String newLine = originalPart + '\t' + new_0.toString() + "\n";
                writer.write(newLine);
                writer.flush();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                reader.close();
            }
            writer.close(); // Close the FileWriter when done
        }
    }
}