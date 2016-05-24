//package hw5;
//
//import weka.core.Instances;
//
//import java.io.BufferedReader;
//import java.io.FileNotFoundException;
//import java.io.FileReader;
//import java.io.IOException;
//
//
//public class MainHW5 {
//    public static void main(String[] args) throws Exception {
//        String training = "src/hw5/ElectionsData_train.txt";
//        String testing = "src/hw5/ElectionsData_test.txt";
//        BufferedReader datafile = readDataFile(training);
//        Instances data = new Instances(datafile);
//        data.setClassIndex(0);
//
//        SVMEval eval = new SVMEval();
//        eval.chooseKernel(data);
//        Instances workingSet = eval.backwardsWrapper(data, 0.05, 5);
//        eval.buildClassifier(workingSet);
//        BufferedReader datafile2 = readDataFile(testing);
//        Instances dataTest = new Instances(datafile2);
//        dataTest.setClassIndex(0);
//        Instances subsetOfFeatures = eval.removeNonSelectedFeatures(dataTest);
//        double avgError = eval.calcAvgError(subsetOfFeatures);
//        System.out.println(avgError);
//    }
//
//    public static BufferedReader readDataFile(String filename) {
//        BufferedReader inputReader = null;
//
//        try {
//            inputReader = new BufferedReader(new FileReader(filename));
//        } catch (FileNotFoundException ex) {
//            System.err.println("File not found: " + filename);
//        }
//
//        return inputReader;
//    }
//}
//
