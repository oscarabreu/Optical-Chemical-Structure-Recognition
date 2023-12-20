import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smiles.SmilesParser;

import java.io.*;
import java.util.ArrayList;

public class CanioncalOne {
    public static void main(String[] args) throws Exception {
        canioncal_one();
}

    public static void canioncal_batch(String fileRead, String fileWrite) throws IOException, InvalidSmilesException {

        File record = new File(fileWrite);//记录结果文件
        FileWriter writer = new FileWriter(record, true);

        File file = new File(fileRead);
        BufferedReader reader = null;
        String temp = null;
        int line = 1;
        try {
            reader = new BufferedReader(new FileReader(file));

            while ((temp = reader.readLine()) != null) {


                try {
                    SmilesParser parser = new SmilesParser(DefaultChemObjectBuilder.getInstance());
                    SmilesGenerator generator = new SmilesGenerator(SmiFlavor.Canonical | SmiFlavor.Isomeric | SmiFlavor.UseAromaticSymbols);
                    String smi_new = generator.createSMILES(parser.parseSmiles(temp));
                    writer.write(smi_new + "\n");
                    line++;
                } catch (Exception e) {
                    System.out.println("出错的SMILES" + line + "\t" + temp);
                    e.printStackTrace();
                }

            }
        } catch (Exception e) {
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

    public static void canioncal_one() throws IOException, InvalidSmilesException {

        SmilesParser parser = new SmilesParser(DefaultChemObjectBuilder.getInstance());

        /**SmiFlavor.Isomeric:会显示锲形键的 加粗的楔形表示处在纸面的外边面向你。无线表示在纸面背后。
         * SmiFlavor.UseAromaticSymbols:会出现那种小的c,只有这个这只为true,生成的图片才会有苯圈，生成图片的代码设置.withAromaticDisplay()属性就可以了。
         */

        SmilesGenerator generator = new SmilesGenerator(SmiFlavor.UseAromaticSymbols | SmiFlavor.Isomeric | SmiFlavor.Canonical | SmiFlavor.CxSmiles);
//        SmilesGenerator generator = new SmilesGenerator(SmiFlavor.CxAtomLabel);

        String[] smi = {"C1=CC(=C(C(=C1CNC2=NCC[F,Cl,Br,I]2)[R1])[R2])[R3]",
                "C1C[F,Cl,Br,I]C(=N1)NCC2=C(C(=CC(=C2)[R3])[R2])[R1]"};

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