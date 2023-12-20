package org.openscience.cdk.renderer.generators.standard;

import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.fingerprint.*;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.smiles.*;
import org.openscience.cdk.silent.*;
import org.openscience.cdk.similarity.*;
import org.xmlcml.cml.base.CC;
import java.io.*;
import java.util.BitSet;
public class TanimotoCalculateOne {
    public static void main(String []args) throws CDKException, IOException {

        SmilesParser smilesParser = new SmilesParser(
                SilentChemObjectBuilder.getInstance()
        );

//        System.out.println(smilesHypethese +"\t"+ smilesTruelabel);
            String smiles1 = "CCCCN(CC)C1=NC(=NC=C12C=C(N(=C2C)[P])(NC3=C(C)C=C(C)C=C3C)[P])C";
            String smiles2 = "CCCCN(C1=NC(C)=NC2=C1C(C)=C([R])[N@]([C@]3=C(C)C=C(C)C=C3C)=C2)CC";
            try{

                IAtomContainer mol1 = smilesParser.parseSmiles(smiles1);
                IAtomContainer mol2 = smilesParser.parseSmiles(smiles2);
                HybridizationFingerprinter fingerprinter = new HybridizationFingerprinter();
                BitSet bitset1 = fingerprinter.getFingerprint(mol1);
                BitSet bitset2 = fingerprinter.getFingerprint(mol2);
                double tanimoto = Tanimoto.calculate(bitset1, bitset2);

                System.out.println ("Tanimoto: "+tanimoto);

            }catch(Exception e){

                System.out.println ("Error！");
            }
        }

    }

