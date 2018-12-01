import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

/*se crea el inicio de sesion en spark */
val spark = SparkSession.builder().getOrCreate()
/*aqui creamos un delimitador (una limpieza) para agrupar los datos con la columna correspondiente en la base de datos*/
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

/* para poder hacer uso de esta base de datos, se necesita tranforma ciertas columna de estrin o numerco*/
val c1 = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val c2 = c1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val  c3= c2.withColumn("y",'y.cast("Int"))


/* k=10 Folding cross Validation */
/* es una técnica utilizada para evaluar los resultados de un análisis estadístico y garantizar que son independientes
de la partición entre datos de entrenamiento y prueba. Consiste en repetir y calcular la media aritmética obtenida de las medidas de evaluación
sobre diferentes particiones. Se utiliza en entornos donde el objetivo principal es la predicción y se quiere estimar la precisión de un modelo que
se llevará a cabo a la práctica.*/

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
val lr = new LinearRegression().setMaxIter(10)
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).addGrid(lr.fitIntercept).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).build()
val trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)
val assembler = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
val assembler1 = (new VectorAssembler().setInputCols(Array("y")).setOutputCol("label"))
val assembler2 = (new VectorAssembler().setInputCols(Array("y")).setOutputCol("prediction"))

val df = assembler.transform(c3)
df.show(5)
val df1 = assembler1.transform(df)
df1.show(5)
val  c55= df1.withColumn("label",'y.cast("Double"))
val Array(trainingData, testData) = c55.randomSplit(Array(0.7,0.3))
val model22 = trainValidationSplit.fit(trainingData)
model22.transform(testData).select("features", "label").show()

/*KMEANS ESTA SECCION */
/*es un algoritmo de clasificación no supervisada (clusterización) que agrupa objetos en k grupos basándose en sus características.
El agrupamiento se realiza minimizando la suma de distancias entre cada objeto y el centroide de su grupo o cluster. Se suele usar la distancia cuadrática.*/

import org.apache.spark.ml.clustering.KMeans
val kmeans = new KMeans().setK(7).setSeed(1L).setPredictionCol("prediction")
val model = kmeans.fit(df)
val categories = model.transform(testData)


/* MULTILAYER PERCEPTRON */
/*  El perceptrón multicapa es una red neuronal artificial (RNA) formada por múltiples capas, de tal manera que tiene capacidad para
resolver problemas que no son linealmente separables, lo cual es la principal limitación del perceptrón (también llamado perceptrón simple). */

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val splits = categories.randomSplit(Array(0.6, 0.4))
val train = splits(0)
val test = splits(1)
val layers = Array[Int](2, 1, 3, 3)
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
val  c4= categories.withColumn("prediction",'prediction.cast("Double"))
val  c5= c4.withColumn("label",'prediction.cast("Double"))
val predictionAndLabels = c5 .select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


/* ARBOL DE DECISIONES */
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor


val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(c55)
val Array(trainingData, testData) = c55.randomSplit(Array(0.7, 0.3))
val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
val model43 = pipeline.fit(trainingData)
val predictions = model.transform(testData)
predictions.select("prediction", "label", "features").show(5)

val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val  e4= predictions.withColumn("prediction",'prediction.cast("Double"))
val rmse = evaluator.evaluate(e4)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")


/* REGRESION LOGISTIC */
/* la regresión logística es un tipo de análisis de regresión utilizado para predecir el resultado de una variable categórica en
función de las variables independientes o predictoras. Es útil para modelar la probabilidad de un evento ocurriendo como función de otros factores.*/

import org.apache.spark.ml.classification.LogisticRegression
val mlr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val mlrModel = mlr.fit(c55)
println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${mlrModel.interceptVector}")
