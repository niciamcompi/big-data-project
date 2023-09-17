import pyspark
from pyspark.sql import SparkSession
import pandas as pd

#create SparkSession
spark = SparkSession.builder\
.master("local[1]")\
.appName("Github")\
.config('spark.ui.port', '4056')\
.getOrCreate()


#import Dataframe
dataset=spark.read.parquet('/ne3.parquet')

#Jahr berechnen
from pyspark.sql.functions import month, year

dataset = dataset.withColumn('year', year(dataset.ts))
dataset.show()

dataset.printSchema()

#create model for imputation with mean
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer

imputed_col = ['UW{}'.format(10+i) for i in range(0, 15)]


model = Imputer(strategy='mean',missingValue=None,inputCols=imputed_col,outputCols=imputed_col).fit(dataset)
impute_data = model.transform(dataset)

#round columns
import pyspark.sql.functions as func
from pyspark.sql.functions import abs

impute_data = impute_data.withColumn("UW10", func.round(impute_data["UW10"], 0))
impute_data = impute_data.withColumn("UW11", func.round(impute_data["UW11"], 0))
impute_data = impute_data.withColumn("UW12", func.round(impute_data["UW12"], 0))
impute_data = impute_data.withColumn("UW13", func.round(impute_data["UW13"], 0))
impute_data = impute_data.withColumn("UW14", func.round(impute_data["UW14"], 0))
impute_data = impute_data.withColumn("UW15", func.round(impute_data["UW15"], 0))
impute_data = impute_data.withColumn("UW16", func.round(impute_data["UW16"], 0))
impute_data = impute_data.withColumn("UW17", func.round(impute_data["UW17"], 0))
impute_data = impute_data.withColumn("UW18", func.round(impute_data["UW18"], 0))
impute_data = impute_data.withColumn("UW19", func.round(impute_data["UW19"], 0))
impute_data = impute_data.withColumn("UW20", func.round(impute_data["UW20"], 0))
impute_data = impute_data.withColumn("UW21", func.round(impute_data["UW21"], 0))
impute_data = impute_data.withColumn("UW22", func.round(impute_data["UW22"], 0))
impute_data = impute_data.withColumn("UW23", func.round(impute_data["UW23"], 0))
impute_data = impute_data.withColumn("UW24", func.round(impute_data["UW24"], 0))
impute_data = impute_data.withColumn("year", func.round(impute_data["year"], 0))

#absolute values
impute_data = impute_data.withColumn("UW10", abs(impute_data.UW10))
impute_data = impute_data.withColumn("UW11", abs(impute_data.UW11))
impute_data = impute_data.withColumn("UW12", abs(impute_data.UW12))
impute_data = impute_data.withColumn("UW13", abs(impute_data.UW13))
impute_data = impute_data.withColumn("UW14", abs(impute_data.UW15))
impute_data = impute_data.withColumn("UW15", abs(impute_data.UW15))
impute_data = impute_data.withColumn("UW16", abs(impute_data.UW16))
impute_data = impute_data.withColumn("UW17", abs(impute_data.UW17))
impute_data = impute_data.withColumn("UW18", abs(impute_data.UW18))
impute_data = impute_data.withColumn("UW19", abs(impute_data.UW19))
impute_data = impute_data.withColumn("UW20", abs(impute_data.UW20))
impute_data = impute_data.withColumn("UW21", abs(impute_data.UW21))
impute_data = impute_data.withColumn("UW22", abs(impute_data.UW22))
impute_data = impute_data.withColumn("UW23", abs(impute_data.UW23))
impute_data = impute_data.withColumn("UW24", abs(impute_data.UW24))

#set all columns as doubletype
impute_data = impute_data.withColumn("UW10",impute_data.UW10.cast('double'))
impute_data = impute_data.withColumn("UW11",impute_data.UW11.cast('double'))
impute_data = impute_data.withColumn("UW12",impute_data.UW12.cast('double'))
impute_data = impute_data.withColumn("UW13",impute_data.UW13.cast('double'))
impute_data = impute_data.withColumn("UW14",impute_data.UW14.cast('double'))
impute_data = impute_data.withColumn("UW15",impute_data.UW15.cast('double'))
impute_data = impute_data.withColumn("UW16",impute_data.UW16.cast('double'))
impute_data = impute_data.withColumn("UW17",impute_data.UW17.cast('double'))
impute_data = impute_data.withColumn("UW18",impute_data.UW18.cast('double'))
impute_data = impute_data.withColumn("UW19",impute_data.UW19.cast('double'))
impute_data = impute_data.withColumn("UW20",impute_data.UW20.cast('double'))
impute_data = impute_data.withColumn("UW21",impute_data.UW21.cast('double'))
impute_data = impute_data.withColumn("UW22",impute_data.UW22.cast('double'))
impute_data = impute_data.withColumn("UW23",impute_data.UW23.cast('double'))
impute_data = impute_data.withColumn("UW24",impute_data.UW24.cast('double'))

#drop timestamp column
impute_data = impute_data.drop("ts")


from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
vectorAssembler = VectorAssembler(inputCols = ['UW10', 'UW11', 'UW12', 'UW13', 'UW14', 'UW15', 'UW16', 'UW17', 'UW18', 'UW19', 'UW20', 'UW21', 'UW22', 'UW23', 'UW24'], outputCol = 'features')
df_vector = vectorAssembler.transform(impute_data)
df_vector.show(5)
df_vector.describe()

indexer = StringIndexer(inputCol = 'year', outputCol = 'label')
df_final = indexer.fit(df_vector).transform(df_vector)
df_final.show(5)
df_final.describe()

#set all columns as doubletype
df_final = df_final.withColumn("UW10",df_final.UW10.cast('double'))
df_final = df_final.withColumn("UW11",df_final.UW11.cast('double'))
df_final = df_final.withColumn("UW12",df_final.UW12.cast('double'))
df_final = df_final.withColumn("UW14",df_final.UW14.cast('double'))
df_final = df_final.withColumn("UW15",df_final.UW15.cast('double'))
df_final = df_final.withColumn("UW16",df_final.UW16.cast('double'))
df_final = df_final.withColumn("UW17",df_final.UW17.cast('double'))
df_final = df_final.withColumn("UW18",df_final.UW18.cast('double'))
df_final = df_final.withColumn("UW19",df_final.UW19.cast('double'))
df_final = df_final.withColumn("UW20",df_final.UW20.cast('double'))
df_final = df_final.withColumn("UW21",df_final.UW21.cast('double'))
df_final = df_final.withColumn("UW22",df_final.UW22.cast('double'))
df_final = df_final.withColumn("UW23",df_final.UW23.cast('double'))
df_final = df_final.withColumn("UW24",df_final.UW24.cast('double'))

df_final.select('year','label').groupBy('year','label').count().show()
import matplotlib.pyplot as plt
#lot = plt.bar(df_final['year', 'label'], yerr=df_final['year', 'label'])

#create train and split dataset
splits = df_final.randomSplit([0.7,0.3])
train_df = splits[0]
test_df = splits[1]
train_df.count(), test_df.count(), df_final.count()


#model training and validation
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

count_train = train_df.count()
col_train = len(train_df.columns)
count_test = test_df.count()
col_test = len(test_df.columns)

layers = [15,1000,8]

a = ['year', 'features']
df_final.select(*a).show()

mlp = MultilayerPerceptronClassifier(featuresCol='features', labelCol='label', layers = layers, seed = 1)

mlp_model = mlp.fit(train_df)
pred_df = mlp_model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'accuracy')
mlpacc = evaluator.evaluate(pred_df)
mlpacc

