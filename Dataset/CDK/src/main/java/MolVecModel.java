import gov.nih.ncats.molvec.Molvec;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.io.MDLV2000Reader;
import org.openscience.cdk.io.MDLV2000Writer;
import org.openscience.cdk.layout.StructureDiagramGenerator;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smiles.SmilesParser;

import java.io.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * @author miracle
 * @Title:
 * @Package
 * @Description:
 * @date 2021/11/521:50
 */
public class MolVecModel {
    public static void main(String[] args) throws IOException, CDKException {
        molConvertSmiles();
    }

    public static void molConvertSmiles() throws IOException, CDKException {
        String dirPath = "F:\\OneDrive - mail.ecust.edu.cn\\毕业论文/基于规则方法对比实验\\";
        String imagesFilePath = "G:\\125WanDataSet\\png";
        String smilesTestFilePath = "F:\\OneDrive - mail.ecust.edu.cn\\毕业论文\\30wanTestDeepSMILES.txt";
        String fileWrite = dirPath + "30wan_molvec_PretectResult.txt";
        int rightNum = 0, errNum = 0, lineNum = 0;
        File record = new File(fileWrite);//记录结果文件
        File image = null;
        FileWriter writer = new FileWriter(record, true);
        BufferedReader enBr = new BufferedReader(new FileReader(smilesTestFilePath));

        while (true) {
            String smilesLable = enBr.readLine();
            if (smilesLable == null) {
                break;
            }

            String[] split = smilesLable.strip().split("\t");
            String num = split[0];
            try {
                File file = new File(imagesFilePath);
                String[] filelist = file.list();
                for (int i = 0; i < filelist.length; i++) {
                    image = new File(imagesFilePath + "\\" + filelist[i] + "\\" + num + ".png");
                    if (image.exists()) {
                        break;
                    } else {
                        continue;
                    }
                }
//                File image =new File( imagesFilePath+num+".png");
                CompletableFuture<String> future = Molvec.ocrAsync(image);
                String ocrMol = future.get(5, TimeUnit.SECONDS);
//                String ocrMol = Molvec.ocr(image);
                MDLV2000Reader mdlr = new MDLV2000Reader(new StringReader(ocrMol));
                IChemObjectBuilder bldr = SilentChemObjectBuilder.getInstance();
                IAtomContainer mol = mdlr.read(bldr.newAtomContainer());
                String smiles = new SmilesGenerator(SmiFlavor.Default).create(mol);
                writer.write(smiles + "\n");
//                System.out.println(smiles);
                rightNum++;
            } catch (Exception e) {
                errNum++;
                writer.write("None" + "\n");
//                System.out.println ("Error！");
            }
        }
        writer.close();
        System.out.println("\nerrorNum:::" + errNum);

    }

    public static void smilesConvertMol() throws CDKException, IOException {
        SmilesParser smilesParser = new SmilesParser(
                DefaultChemObjectBuilder.getInstance()
        );

        String smiles = "CCC(O)C";
        try {
            IAtomContainer molecule = smilesParser.parseSmiles(smiles);
            StructureDiagramGenerator sdg = new StructureDiagramGenerator();
            sdg.setMolecule(molecule);
            sdg.generateCoordinates(molecule);
            molecule = sdg.getMolecule();
            MDLV2000Writer writer = new MDLV2000Writer(
                    new FileWriter(new File("C:\\Users\\Administrator\\Desktop\\output.mol"))
            );
            writer.write(molecule);
            writer.close();
        } catch (InvalidSmilesException | IOException e) {
            System.err.println(e.getMessage());
        } catch (CDKException e) {
            e.printStackTrace();
        }

    }

}
