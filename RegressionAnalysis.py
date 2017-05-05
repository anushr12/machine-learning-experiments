from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors as MLLibVectors
# data = spark.read.format("libsvm").load("s3a://log.alikeaudience.com/NextBuzz_Activity/TrainingDataNextBuzz.txt")

# lines = sc.textFile("s3a://log.alikeaudience.com/NextBuzz_Activity/TrainingDataNextBuzz.txt")
# parts = lines.map(lambda l: l.split(" "))
# # Each line is converted to a tuple.
# people = parts.map(lambda p: (p[4], [p[0],p[1],p[2],p[3]]))

# # The schema is encoded in a string.
# schemaString = "label features"

# fields = [StructField(field_name, ArrayType(FloatType(),False), False) for field_name in schemaString.split()]
# schema = StructType(fields)

# # Apply the schema to the RDD.
# schemaPeople = spark.createDataFrame(people, schema)

# labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# schemaPeople.show()

# lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

lr = LogisticRegression(maxIter=5, regParam=0.01)
# lr = LogisticRegressionWithSGD.train(iterations=10)

# Fit the model
lrModel = lr.fit(inputDF)
lrcol=lr.getFeaturesCol()

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

result = lrModel.transform(testDF).head()
# result.prediction
# test1 = sc.parallelize([Row(features=MLLibVectors.dense([2,102,30,4.385]))]).toDF()
test1 = sc.parallelize([Row(features=MLLibVectors.dense([2,173,30,8.281666667]))]).toDF()

result1 = lrModel.transform(test1)
print result1.collect()
print(result1.prediction)




import numpy as np
from pyspark.ml.linalg import Vectors as MLLibVectors

linesTrain = sc.textFile("s3a://log.alikeaudience.com/NextBuzz_Activity/TrainingDataNextBuzz.txt")

inputD = []

def CreateDF(lines):
    for each in lines.collect():
        eachSplit = each.split(" ")
        label = int(eachSplit[4])
        features = []
        for feature in eachSplit[:4]:
            features.append(feature)
        inputD.append(Row(label=label, features=MLLibVectors.dense(features)))
    return inputD

inputDF= sc.parallelize(CreateDF(linesTrain)).toDF()

testD = []
def CreateDFTest(lines):
    for each in lines.collect():
        eachSplit = each.split(" ")
        # label = int(eachSplit[4])
        features = []
        for feature in eachSplit[:4]:
            features.append(feature)
        testD.append(Row(features=MLLibVectors.dense(features)))
    return testD
# inputDF.first()
# inputDF.show()

linesTest = sc.textFile("s3a://log.alikeaudience.com/NextBuzz_Activity/TestDataNextBuzz_eval.txt")

testDF = sc.parallelize(CreateDFTest(linesTest)).toDF()
testDF.show()