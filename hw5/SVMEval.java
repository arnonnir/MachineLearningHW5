package hw5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;


import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SVMEval {
	final static int NUM_FOLDS = 3;
	public SMO m_SMO;
	private ArrayList<Integer> m_setOfAttributeIndexes;

	public SVMEval() {
		m_SMO = new SMO();
	}

	public void buildClassifier(Instances instances) throws Exception {
		m_SMO.buildClassifier(instances);
	}

	public void chooseKernel(Instances instances) throws Exception {
		double minPolynomialKernelError = Double.MAX_VALUE;
		double minRBFKernelError = Double.MAX_VALUE;
		int indexOfMinPolynomialKernel = 2;
		int indexOfMinRBFKernel = -10;
		PolyKernel polyKernel = new PolyKernel();
		RBFKernel rbfKernel = new RBFKernel();

		for(int i = 2; i <= 4; i++) {
			polyKernel.setExponent(i);
			m_SMO.setKernel(polyKernel);
			double tempError = calcCrossValidationError(instances);

			if(tempError < minPolynomialKernelError) {
				minPolynomialKernelError = tempError;
				indexOfMinPolynomialKernel = i;
			}
		}

		polyKernel.setExponent(indexOfMinPolynomialKernel);

		for(int i = -10; i <= -2; i++) {
			rbfKernel.setGamma(Math.pow(2, i));
			m_SMO.setKernel(rbfKernel);
			double tempError = calcCrossValidationError(instances);

			if(tempError < minRBFKernelError) {
				minRBFKernelError = tempError;
				indexOfMinRBFKernel = i;
			}
		}

		rbfKernel.setGamma(Math.pow(2, indexOfMinRBFKernel));

		if(minPolynomialKernelError < minRBFKernelError) {
			m_SMO.setKernel(polyKernel);
		}else {
			m_SMO.setKernel(rbfKernel);
		}
	}

	public double calcCrossValidationError(Instances instances) throws Exception {
		// shuffle instances before divide them
		Random random = new Random();
		instances.randomize(random);
		double crossValidationError = 0;

		int numOfInstances = instances.numInstances();
		// if the number of instances is lower then num of folds (3),
		// assign num of folds to be the number of instances
		int numOfFolds = (numOfInstances < NUM_FOLDS) ? numOfInstances : NUM_FOLDS;
		for (int n = 0; n < numOfFolds; n++) {
			// these methods divide the data so that fold n is the testing data, and the rest is the training
			Instances testingSet = instances.testCV(numOfFolds, n);
			Instances trainingSet = instances.trainCV(numOfFolds, n);
			buildClassifier(trainingSet);
			double specificFoldError = calcAvgError(testingSet);
			crossValidationError += specificFoldError;
		}

		crossValidationError /= (double) numOfFolds;

		return crossValidationError;
	}



	public double calcAvgError(Instances testingData) {
		int numOfFoldInstances = testingData.numInstances();
		double totalFoldError = 0;

		for (int i = 0; i < numOfFoldInstances; i++) { // run throw all instances in the specific fold
			double classValue = testingData.instance(i).classValue();
			double predictedValue;
			try {
				predictedValue = m_SMO.classifyInstance(testingData.instance(i));
				totalFoldError += (predictedValue != classValue) ? 1 : 0;
			}catch(Exception e) {
				System.out.println("There is a problem to classify the instance in index: " + i);
			}
		}

		double avgFoldError = totalFoldError / (double)numOfFoldInstances;

		return avgFoldError;
	}

	public Instances backwardsWrapper(Instances instances, double threshold, int minNumberOfAttributes) throws Exception {
		double errorDiff = 0;
		double originalError = calcCrossValidationError(instances);

		m_setOfAttributeIndexes = new ArrayList<Integer>();

		while(instances.numAttributes() - 1 > minNumberOfAttributes && errorDiff < threshold) {
			int indexOfMinimalAttribute = 1;
			Instances dataWithoutAtt = removeAttributes(instances, 1);
			double minimalError = calcCrossValidationError(dataWithoutAtt);

			for (int i = 2; i <= instances.numAttributes() - 1; i++) {
				dataWithoutAtt = removeAttributes(instances, i);
				double newError = calcCrossValidationError(dataWithoutAtt);

				if(newError < minimalError) {
					minimalError = newError;
					indexOfMinimalAttribute = i;
				}
			}

			errorDiff = minimalError - originalError;

			if(errorDiff < threshold) {
				instances = removeAttributes(instances, indexOfMinimalAttribute);
				m_setOfAttributeIndexes.add(indexOfMinimalAttribute);
			}
		}

		return instances;
	}


	public Instances removeNonSelectedFeatures(Instances dataSet) throws Exception {
		for(int i = 0; i < m_setOfAttributeIndexes.size(); i++) {
			dataSet = removeAttributes(dataSet, m_setOfAttributeIndexes.get(i));
		}

		return dataSet;
	}

	private Instances removeAttributes(Instances instances, int index) throws Exception {
		Remove remove = new Remove();
		remove.setInputFormat(instances);
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = Integer.toString(index + 1);
		remove.setOptions (options);
		Instances workingSet = Filter.useFilter(instances, remove);

		return workingSet;
	}

	public static void main(String[] args) throws Exception {
		String training = "src/hw5/ElectionsData_train.txt";
		String testing = "src/hw5/ElectionsData_test.txt";
		BufferedReader datafile = readDataFile(training);
		Instances data = new Instances(datafile);
		data.setClassIndex(0);

		SVMEval eval = new SVMEval();
		eval.chooseKernel(data);
		Instances workingSet = eval.backwardsWrapper(data, 0.05, 5);
		eval.buildClassifier(workingSet);
		BufferedReader datafile2 = readDataFile(testing);
		Instances dataTest = new Instances(datafile2);
		dataTest.setClassIndex(0);
		Instances subsetOfFeatures = eval.removeNonSelectedFeatures(dataTest);
		double avgError = eval.calcAvgError(subsetOfFeatures);
		System.out.println(avgError);
	}

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

}
