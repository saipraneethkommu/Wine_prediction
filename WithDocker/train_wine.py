from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, MulticlassMetrics
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def clean_data(df):
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

def main():
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    entry_path = "TrainingDataset.csv"
    correct_path = "ValidationDataset.csv"
    # output_path = "s3://awsbucketwine/output"
    print(f"Reading training CSV file from {entry_path}")
    train_df = (
        spark.read.format("csv")
        .option("header", True)
        .option("sep", ";")
        .option("inferschema", True)
        .load(entry_path)
    )
    train_df = clean_data(train_df)
    print(f"Reading validation CSV file from {correct_path}")

    valid_df = (
        spark.read.format("csv")
        .option("header", True)
        .option("sep", ";")
        .option("inferschema", True)
        .load(correct_path)
    )

    valid_df = clean_data(valid_df)
    all_features = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    assembler = VectorAssembler(inputCols=all_features, outputCol="features")
    indexer = StringIndexer(inputCol="quality", outputCol="label")
    train_df.cache()
    valid_df.cache()
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=150,
        maxDepth=15,
        seed=150,
        impurity="gini",
    )

    pipeline = Pipeline(stages=[assembler, indexer, rf])
    model = pipeline.fit(train_df)
    predictions = model.transform(valid_df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)
    print(f"Test Accuracy of wine prediction model = {accuracy}")

    metrics = MulticlassMetrics(predictions.rdd.map(tuple))
    print(f"Weighted f1 score of wine prediction model = {metrics.weightedFMeasure()}")

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [6, 9]) \
        .addGrid(rf.numTrees, [50, 150]) \
        .addGrid(rf.minInstancesPerNode, [6]) \
        .addGrid(rf.seed, [100, 200]) \
        .addGrid(rf.impurity, ["entropy", "gini"]) \
        .build()

    pipeline = Pipeline(stages=[assembler, indexer, rf])
    crossval = CrossValidator(
        estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=2
    )
    cvmodel = crossval.fit(train_df)

    predictions = cvmodel.transform(valid_df)
    results = predictions.select(["prediction", "label"])
    accuracy = evaluator.evaluate(predictions)
    print(
        f"Test Accuracy of wine prediction model (after CrossValidation) = {accuracy}"
    )

    metrics = MulticlassMetrics(results.rdd.map(tuple))

main()
