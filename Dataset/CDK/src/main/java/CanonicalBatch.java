import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smiles.SmilesParser;

import java.io.*;
import java.util.ArrayList;

public class CanonicalBatch {
    public static void main(String[] args) throws Exception {
        System.out.println("Starting CanonicalBatch...");
        String dirPath = "125wanKS";
        String fileRead = dirPath + ".txt";
        String fileWrite = dirPath + "_Canonical.txt";
        System.out.println("Reading from: " + fileRead);
        System.out.println("Writing to: " + fileWrite);
        canonical_batch(fileRead, fileWrite);
        System.out.println("CanonicalBatch completed.");
    }

   public static void canonical_batch(String fileRead, String fileWrite) throws IOException, InvalidSmilesException {
        File record = new File(fileWrite);
        FileWriter writer = new FileWriter(record, true);

        File file = new File(fileRead);
        BufferedReader reader = null;
        String temp = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            while ((temp = reader.readLine()) != null) {
                try {
                    SmilesParser parser = new SmilesParser(DefaultChemObjectBuilder.getInstance());
                    SmilesGenerator generator = new SmilesGenerator(SmiFlavor.Canonical | SmiFlavor.Isomeric | SmiFlavor.UseAromaticSymbols);

                    String[] sequence = temp.split("\t");
                    String smilesCanonical;
                    if (sequence.length == 1) {
                        smilesCanonical = generator.create(parser.parseSmiles(sequence[0]));
                        writer.write(smilesCanonical + "\n");
                    } else {
                        smilesCanonical = generator.create(parser.parseSmiles(sequence[1]));
                        writer.write(sequence[0] + "\t" + smilesCanonical + "\n");
                    }
                } catch (Exception e) {
                    writer.write(temp + "\n"); // Write the original SMILES string
                    e.printStackTrace();
                }
            }
        } catch (Exception e) {
            writer.write("Error in reading file\n");
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            if (writer != null) {
                try {
                    writer.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void canonical_one() throws IOException, InvalidSmilesException {

        SmilesParser parser = new SmilesParser(DefaultChemObjectBuilder.getInstance());
        SmilesGenerator generator = new SmilesGenerator(SmiFlavor.UseAromaticSymbols | SmiFlavor.Isomeric | SmiFlavor.Canonical | SmiFlavor.CxSmiles);
        String[] smi = {"C1=C(C=C(C(=C1)))C(=O)",
                "C1=C(C=C(C(=C1)[R20])Br)C(=O)[Rm]"};

        ArrayList<String> can = new ArrayList<String>();

        for (String s : smi) {
            can.add(generator.createSMILES(parser.parseSmiles(s)));
        }
        for (String s : can) {
            System.out.println(s);
        }
        if (can.get(0).equals(can.get(1))) {
            System.out.println("==");
        } else {
            System.out.println("!=");
        }
    }
}