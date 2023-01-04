// Copyright Header - Semi-Supervised k-Star (SSS) with Holo-Training
// Copyright (C) 2023 Kokten Ulas BIRANT

// Copyright is owned by the author working at Dokuz Eylul University. 
// You can use the algorithm for academic and research purposes only, e.g. not for commercial use, without a fee. 

using System;
using weka.core;
using weka.classifiers.lazy;

namespace SSS
{
    class Program
    {
        public static double Training(string datasetName, int folds)
        {
            Instances insts = new Instances(new java.io.FileReader("datasets\\" + datasetName + ".arff"));  // Read dataset file

            java.util.Random rand = new java.util.Random(2);  // Run n times with different random seeds and then find average accuracy 
            insts.randomize(rand);

            insts.setClassIndex(insts.numAttributes() - 1);     // The last attribute is the target attribute, which contains class labels 
            if (insts.classAttribute().isNominal())             // Tests whether class attribute is nominal 
                insts.stratify(folds);
                  
            KStar pre_classifier = new KStar();      // classifier 1
            KStar pseudo_classifier = new KStar();   // classifier 2 
            KStar post_classifier = new KStar();     // classifier 3

            double accuracy = 0;

            for (int n = 0; n < folds; n++)
            {
                Instances trainData = insts.trainCV(folds, n, new java.util.Random(2));  // Run n times with different random seeds and then find average accuracy
                Instances testData = insts.testCV(folds, n);

                double percentSplit = 75;       // Labeled data percentage  

                int labeledSize = Convert.ToInt16(Convert.ToDecimal(Math.Round(trainData.numInstances() * percentSplit / 100)));

                Instances labeledData = new Instances(trainData, 0, labeledSize);
           
                pre_classifier.buildClassifier(labeledData);   // Construct pre-classifier

                // Labelling unlabeled data   
                for (int i = labeledSize; i < trainData.numInstances(); i++)
                {
                    double predictedClass = pre_classifier.classifyInstance(trainData.instance(i));  // Make prediction
                    trainData.instance(i).setClassValue(predictedClass);
                }

                post_classifier.buildClassifier(trainData);  /// Construct post-classifier 

                Instances pseudoLabeledData = new Instances(trainData, labeledSize, (trainData.numInstances() - labeledSize));
                pseudo_classifier.buildClassifier(pseudoLabeledData);   // construct pseudo-classifier 

                double correct = 0;

                for (int i = 0; i < testData.numInstances(); i++)
                {
                    double predictedClass1 = pre_classifier.classifyInstance(testData.instance(i));     // prediction of classifier 1
                    double predictedClass2 = pseudo_classifier.classifyInstance(testData.instance(i));  // prediction of classifier 2 
                    double predictedClass3 = post_classifier.classifyInstance(testData.instance(i));    // prediction of classifier 3  

                    double finalPredictedClass = predictedClass1;
                    if ((predictedClass1 != predictedClass2) && (predictedClass1 != predictedClass3) && (predictedClass2 != predictedClass3))
                        finalPredictedClass = predictedClass1;
                    else if ((predictedClass1 == predictedClass2) || (predictedClass1 == predictedClass3))
                        finalPredictedClass = predictedClass1;
                    else if (predictedClass2 == predictedClass3)
                        finalPredictedClass = predictedClass2;
                    else
                        finalPredictedClass = predictedClass3;

                    if (finalPredictedClass == testData.instance(i).classValue())    // Compare predicted and actual class labels
                        correct++;
                }
                accuracy = accuracy + Math.Round((correct / testData.numInstances()), 4) * 100;

            }
            return accuracy / folds;
        }
        
        static void Main(string[] args)
        {
            string[] datasetNames = { "d1", "d2", "d3" };   // Dataset names

            Console.WriteLine("Results");
                        
            for (int i = 0; i < datasetNames.Length; i++)
                 Console.WriteLine(Training(datasetNames[i], 10));   // run with two parameters: (i) dataset name and (ii) the number of folds for cross validation 
                       
            Console.ReadKey();
        }
    }
}

